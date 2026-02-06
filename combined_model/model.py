# combined_model/model.py
"""
Combined Semantic ViT - Dual pathway model

Fuses both ViT (sequential) and RvNN (structural) paths:

    Code -> CodeBERT -> PixelEmbedder
        ├── PatchEmbedder -> Projector       -> [M, 1536]  (ViT path)
        └── PixelAdapter  -> RecursiveEncoder -> [1, 1536]  (Tree path)
                            ↓
                 cat([global, seq]) -> [M+1, 1536]
                            ↓
                       QwenDecoder
"""
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

        # --- PATH 1: SEQUENTIAL (ViT) ---
        self.patch_embedder = PatchEmbedder(patch_size=patch_size)
        self.projector = Projector(
            in_dim=patch_size * 768,
            bottleneck_dim=bottleneck_dim,
            out_dim=1536,
            dropout=dropout
        ).to(DEVICE)

        # --- PATH 2: STRUCTURAL (RvNN) ---
        self.pixel_adapter = nn.Sequential(
            nn.Linear(768, 1536),
            nn.LayerNorm(1536),
            nn.GELU()
        ).to(DEVICE)

        self.recursive_encoder = RecursiveEncoder(
            embed_dim=1536,
            max_branching=8,
            hidden_dim=3072,
            dropout=dropout
        ).to(DEVICE)

        # --- DECODER ---
        self.decoder = QwenDecoder(device=DEVICE)

    def get_trainable_parameters(self):
        """Return all trainable parameters from both paths."""
        params = []
        # Shared
        params.extend(self.pixel_embedder.parameters())
        # ViT path
        params.extend(self.projector.parameters())
        # Tree path
        params.extend(self.pixel_adapter.parameters())
        params.extend(self.recursive_encoder.parameters())
        return params

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
        # [N, 768] -> [M, 3072] -> [M, 1536]
        patch_embeddings = self.patch_embedder(pixel_embeddings)
        seq_vectors = self.projector(patch_embeddings)

        # --- PATH 2: Structural ---
        # [N, 768] -> [1, N, 1536] -> [1, 1536]
        pixels_for_tree = self.pixel_adapter(pixel_embeddings).unsqueeze(0)
        global_vector = self.recursive_encoder.forward_tree(
            features['tree_roots'],
            pixels_for_tree
        )

        # --- FUSION ---
        # [1, 1536] + [M, 1536] -> [M+1, 1536]
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
