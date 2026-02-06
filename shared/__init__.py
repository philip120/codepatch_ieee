# shared/__init__.py
"""
Shared components used by model/, model2/, and combined_model/.

Components:
    - SemanticExtractor: Extracts semantic operations from MATLAB code
    - CodeBERTEncoder: Frozen CodeBERT for CLS embeddings
    - PixelEmbedder: Adds depth + type embeddings to CLS
    - PatchEmbedder: Groups pixels into patches (ViT-style)
    - Projector: Bottleneck MLP to Qwen embedding space
    - QwenDecoder: Frozen Qwen LLM for text generation

Constants:
    - TYPE_TO_ID, ID_TO_TYPE, NUM_TYPES, MAX_DEPTH
"""

from .semantic_extractor import SemanticExtractor, TYPE_TO_ID, ID_TO_TYPE, NUM_TYPES, MAX_DEPTH
from .codebert_encoder import CodeBERTEncoder
from .pixel_embedder import PixelEmbedder
from .patch_embedder import PatchEmbedder
from .projector import Projector
from .qwen_decoder import QwenDecoder
