"""
Structural Model (Model 2) - Tree-only pipeline

Structural path only:
    Code -> SemanticExtractorV2 -> CodeBERT -> PixelEmbedder -> PixelAdapter -> RecursiveEncoder -> QwenDecoder
"""
import time
import torch
import torch.nn as nn

from .semantic_extractor import SemanticExtractorV2
from shared.codebert_encoder import CodeBERTEncoder
from shared.pixel_embedder import PixelEmbedder
from shared.semantic_extractor import MAX_DEPTH, NUM_TYPES
from shared.qwen_decoder import QwenDecoder
from .recursive_encoder import RecursiveEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StructuralModel(nn.Module):
    """
    Tree-only model for code-to-pseudocode.

    Uses recursive tree encoding of the AST to produce a global vector.

    Trainable: PixelEmbedder, PixelAdapter, RecursiveEncoder
    Frozen: CodeBERTEncoder, QwenDecoder
    """

    def __init__(
        self,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Extractor (tree-aware)
        self.extractor = SemanticExtractorV2()

        # Base Encoder (frozen)
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # Pixel Embedder (trainable)
        self.pixel_embedder = PixelEmbedder(
            max_depth=MAX_DEPTH,
            num_types=NUM_TYPES
        ).to(DEVICE)

        # Adapt pixel (768) to Recursive dim (1536)
        self.pixel_adapter = nn.Sequential(
            nn.Linear(768, 1536),
            nn.LayerNorm(1536),
            nn.GELU()
        ).to(DEVICE)

        # Recursive Encoder (tree aggregation)
        self.recursive_encoder = RecursiveEncoder(
            embed_dim=1536,
            max_branching=8,
            hidden_dim=3072,
            dropout=dropout
        ).to(DEVICE)

        # Decoder (frozen)
        self.decoder = QwenDecoder(device=DEVICE)

    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        params = []
        params.extend(self.pixel_embedder.parameters())
        params.extend(self.pixel_adapter.parameters())
        params.extend(self.recursive_encoder.parameters())
        params.extend(self.decoder.get_lora_parameters())
        return params

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

        # 3. Pixel Embeddings [N, 768]
        pixel_embeddings = self.pixel_embedder(cls_embeddings, depth_ids, type_ids)

        # 4. Adapt for tree: [N, 768] -> [1, N, 1536]
        pixels_for_tree = self.pixel_adapter(pixel_embeddings).unsqueeze(0)

        # 5. Recursive tree traversal -> [1, 1536]
        global_vector = self.recursive_encoder.forward_tree(
            features['tree_roots'],
            pixels_for_tree
        )

        # 6. Decode
        if target:
            return self.decoder.forward_train(global_vector, target)
        else:
            return global_vector

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
