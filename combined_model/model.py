# combined_model/model.py
"""
Combined Semantic ViT - Dual pathway model

Fuses both ViT (sequential) and RvNN (structural) paths:

    Code -> CodeBERT -> PixelEmbedder
        ├── PatchEmbedder -> Projector       -> [M, qwen_dim]  (ViT path)
        └── PixelAdapter  -> RecursiveEncoder -> [1, qwen_dim]  (Tree path)
                            ↓
                 cat([global, seq]) -> [M+1, qwen_dim]
                            ↓
                       QwenDecoder
"""
import time
import torch
import torch.nn as nn

from model2.semantic_extractor import SemanticExtractorV2
from shared.codebert_encoder import CodeBERTEncoder
from shared.pixel_embedder import PixelEmbedder
from shared.patch_embedder import PatchEmbedder
from shared.projector import Projector
from shared.semantic_extractor import MAX_DEPTH, NUM_TYPES
from shared.qwen_decoder import QwenDecoder
from model2.recursive_encoder import RecursiveEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CombinedSemanticViT(nn.Module):
    """
    Combined model with both ViT and Tree pathways.

    Trainable:
        - PixelEmbedder (shared between paths)
        - PatchEmbedder + Projector (ViT path)
        - PixelAdapter + RecursiveEncoder (Tree path)

    Frozen:
        - CodeBERTEncoder
        - QwenDecoder
    """

    def __init__(
        self,
        patch_size: int = 4,
        bottleneck_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Extractor (tree-aware, needed for both paths)
        self.extractor = SemanticExtractorV2()

        # Base Encoder (frozen)
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # Pixel Embedder (trainable, shared)
        self.pixel_embedder = PixelEmbedder(
            max_depth=MAX_DEPTH,
            num_types=NUM_TYPES
        ).to(DEVICE)

        # --- DECODER (created first so we can read hidden_size) ---
        self.decoder = QwenDecoder(device=DEVICE)
        qwen_dim = self.decoder.hidden_size

        # --- PATH 1: SEQUENTIAL (ViT) ---
        self.patch_embedder = PatchEmbedder(patch_size=patch_size)
        self.projector = Projector(
            in_dim=patch_size * 768,
            bottleneck_dim=bottleneck_dim,
            out_dim=qwen_dim,
            dropout=dropout
        ).to(DEVICE)

        # --- PATH 2: STRUCTURAL (RvNN) ---
        # No LayerNorm: same reason as projector — pins norm to sqrt(qwen_dim)≈50.
        # Kaiming init on Linear(768, qwen_dim) gives output norm ≈ sqrt(2*qwen_dim/768) ≈ 2.6,
        # which is close enough to Qwen token norms (~1.09).
        self.pixel_adapter = nn.Linear(768, qwen_dim).to(DEVICE)

        self.recursive_encoder = RecursiveEncoder(
            embed_dim=qwen_dim,
            max_branching=8,
            hidden_dim=qwen_dim * 2,
            dropout=dropout
        ).to(DEVICE)


    def get_trainable_parameters(self):
        """Return all encoder trainable parameters (excludes unfrozen Qwen layers)."""
        params = []
        # Shared
        params.extend(self.pixel_embedder.parameters())
        # ViT path
        params.extend(self.projector.parameters())
        # Tree path
        params.extend(self.pixel_adapter.parameters())
        params.extend(self.recursive_encoder.parameters())
        # LoRA (if enabled; empty list when using unfreeze)
        params.extend(self.decoder.get_lora_parameters())
        return params

    def num_trainable_parameters(self):
        """Total trainable parameters including unfrozen Qwen layers."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_lora(self, **kwargs):
        """Enable LoRA on the Qwen decoder."""
        self.decoder.enable_lora(**kwargs)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.decoder.train_mode()
        else:
            self.decoder.eval_mode()
        return self

    def eval(self):
        return self.train(False)

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.get_trainable_parameters())

    def forward(self, code: str, target: str = None, features: dict = None):
        # 1. Extract features (with tree structure)
        if features is None:
            features = self.extractor(code)

        if not features['texts']:
            if target:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return None

        # 2. Base Embeddings [N, 768]
        cls_embeddings = self.encoder(features['texts'])

        depth_ids = torch.tensor(features['depths'], device=DEVICE)
        type_ids = torch.tensor(features['type_ids'], device=DEVICE)

        # 3. Pixel Embeddings [N, 768] (shared)
        pixel_embeddings = self.pixel_embedder(cls_embeddings, depth_ids, type_ids)

        # --- PATH 1: Sequential ---
        # [N, 768] -> [M, patch*768] -> [M, qwen_dim]
        patch_embeddings = self.patch_embedder(pixel_embeddings)
        seq_vectors = self.projector(patch_embeddings)

        # --- PATH 2: Structural ---
        # [N, 768] -> [1, N, qwen_dim] -> [1, qwen_dim]
        pixels_for_tree = self.pixel_adapter(pixel_embeddings).unsqueeze(0)
        global_vector = self.recursive_encoder.forward_tree(
            features['tree_roots'],
            pixels_for_tree
        )

        # --- FUSION ---
        # Normalise global_vector to unit norm, same as projector output.
        global_vector = torch.nn.functional.normalize(global_vector, dim=-1)

        # [1, qwen_dim] + [M, qwen_dim] -> [M+1, qwen_dim]
        combined = torch.cat([global_vector, seq_vectors], dim=0)

        # --- DECODE ---
        if target:
            return self.decoder.forward_train(combined, target)
        else:
            return combined

    @torch.no_grad()
    def generate(self, code: str, max_new_tokens: int = 128):
        projected = self.forward(code)
        if projected is None:
            return ""
        return self.decoder.generate(projected, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def generate_with_metrics(self, code: str, max_new_tokens: int = 128):
        """Generate text and return (text, efficiency_metrics)."""
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        projected = self.forward(code)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if projected is None:
            return "", {}

        text, dec_metrics = self.decoder.generate_with_metrics(
            projected, max_new_tokens=max_new_tokens
        )

        peak_vram_mb = 0.0
        if DEVICE == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        return text, {
            "encode_time_s": round(t1 - t0, 4),
            "total_time_s": round(time.perf_counter() - t0, 4),
            "peak_vram_mb": round(peak_vram_mb, 1),
            **dec_metrics,
        }
