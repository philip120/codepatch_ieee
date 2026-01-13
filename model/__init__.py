# model/__init__.py
"""
Semantic ViT Model Components

Structure:
    MATLAB Code
        │
        ▼ SemanticExtractor (Step 1)
    [texts, depths, type_ids]
        │
        ▼ CodeBERTEncoder (Step 2)
    CLS embeddings [N, 768]
        │
        ▼ PixelEmbedder (Step 3)
    pixel_embeddings [N, 768]
        │
        ▼ PatchEmbedder (Step 4)
    patch_embeddings [N/P, P*768]
        │
        ▼ Projector (Step 5)
    projected [N/P, 1536]
        │
        ▼ Qwen (Step 6)
    output text
"""

from .semantic_extractor import SemanticExtractor, TYPE_TO_ID, NUM_TYPES, MAX_DEPTH
from .codebert_encoder import CodeBERTEncoder
from .pixel_embedder import PixelEmbedder
from .patch_embedder import PatchEmbedder
from .projector import Projector
from .qwen_decoder import QwenDecoder
