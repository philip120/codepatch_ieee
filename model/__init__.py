# model/__init__.py
"""
Semantic ViT Model (ViT-only pipeline)

Structure:
    MATLAB Code
        │
        ▼ SemanticExtractor
    [texts, depths, type_ids]
        │
        ▼ CodeBERTEncoder
    CLS embeddings [N, 768]
        │
        ▼ PixelEmbedder
    pixel_embeddings [N, 768]
        │
        ▼ PatchEmbedder
    patch_embeddings [N/P, P*768]
        │
        ▼ Projector
    projected [N/P, 1536]
        │
        ▼ Qwen
    output text
"""

from shared.semantic_extractor import SemanticExtractor, TYPE_TO_ID, NUM_TYPES, MAX_DEPTH
from shared.codebert_encoder import CodeBERTEncoder
from shared.pixel_embedder import PixelEmbedder
from shared.patch_embedder import PatchEmbedder
from shared.projector import Projector
from shared.qwen_decoder import QwenDecoder
