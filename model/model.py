# model/model.py
"""
Semantic ViT (Model 1) - ViT-only pipeline

Sequential path only:
    Code -> SemanticExtractor -> CodeBERT -> PixelEmbedder -> PatchEmbedder -> Projector -> QwenDecoder
"""
import torch
import torch.nn as nn

from shared.semantic_extractor import SemanticExtractor, MAX_DEPTH, NUM_TYPES
from shared.codebert_encoder import CodeBERTEncoder
from shared.pixel_embedder import PixelEmbedder
from shared.patch_embedder import PatchEmbedder
from shared.projector import Projector
from shared.qwen_decoder import QwenDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticViT(nn.Module):
    """
    ViT-only model for code-to-pseudocode.

    Trainable: PixelEmbedder, Projector
    Frozen: CodeBERTEncoder, QwenDecoder
    """

    def __init__(
        self,
        patch_size: int = 4,
        bottleneck_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.patch_size = patch_size

        # Extractor (not nn.Module)
        self.extractor = SemanticExtractor()

        # CodeBERT (frozen, not nn.Module)
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # Pixel embedder (TRAINABLE)
        self.pixel_embedder = PixelEmbedder(
            max_depth=MAX_DEPTH,
            num_types=NUM_TYPES,
        ).to(DEVICE)

        # Patch embedder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size)

        # Projector (TRAINABLE)
        self.projector = Projector(
            in_dim=patch_size * 768,
            bottleneck_dim=bottleneck_dim,
            out_dim=1536,
            dropout=dropout,
        ).to(DEVICE)

        # Qwen decoder (frozen, not nn.Module)
        self.decoder = QwenDecoder(device=DEVICE)

    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        params = []
        params.extend(self.pixel_embedder.parameters())
        params.extend(self.projector.parameters())
        return params

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.get_trainable_parameters())

    def forward(self, code: str, target: str = None, features: dict = None):
        # 1. Extract semantic features
        if features is None:
            features = self.extractor(code)

        if not features['texts']:
            if target:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return None

        # 2. CodeBERT embeddings [N, 768]
        cls_embeddings = self.encoder(features['texts'])

        # 3. Pixel embeddings [N, 768]
        depth_ids = torch.tensor(features['depths'], device=DEVICE)
        type_ids = torch.tensor(features['type_ids'], device=DEVICE)
        pixel_embeddings = self.pixel_embedder(cls_embeddings, depth_ids, type_ids)

        # 4. Patch embeddings [M, 3072]
        patch_embeddings = self.patch_embedder(pixel_embeddings)

        # 5. Project to Qwen space [M, 1536]
        projected = self.projector(patch_embeddings)

        # 6. Decode
        if target:
            return self.decoder.forward_train(projected, target)
        else:
            return projected

    @torch.no_grad()
    def generate(self, code: str, max_new_tokens: int = 128) -> str:
        """Generate pseudocode from MATLAB code."""
        projected = self.forward(code, target=None)
        if projected is None:
            return ""
        return self.decoder.generate(projected, max_new_tokens=max_new_tokens)
