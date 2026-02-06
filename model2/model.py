"""
Hybrid Semantic ViT (Model 2)

Combines:
1. Sequential Path (ViT): CodeBERT -> PixelEmb -> PatchEmb -> Projector -> [Sequence]
2. Structural Path (RvNN): CodeBERT -> PixelEmb -> TreeStructure -> RecursiveEncoder -> [Global]

Fusion:
[Global Vector] concatenated with [Sequence Vectors] -> Qwen Decoder
"""
import torch
import torch.nn as nn

from .semantic_extractor import SemanticExtractorV2
from .codebert_encoder import CodeBERTEncoder
from model.pixel_embedder import PixelEmbedder, MAX_DEPTH, NUM_TYPES # Reuse from model 1
from model.patch_embedder import PatchEmbedder # Reuse from model 1
from .projector import Projector
from .recursive_encoder import RecursiveEncoder
from .qwen_decoder import QwenDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HybridSemanticViT(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        bottleneck_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        # 1. Extractor (Tree-Aware)
        self.extractor = SemanticExtractorV2()
        
        # 2. Base Encoder (Frozen)
        self.encoder = CodeBERTEncoder(device=DEVICE)
        
        # 3. Pixel Embedder (Trainable)
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
        self.recursive_encoder = RecursiveEncoder(
            embed_dim=1536, # Output dim matches Qwen
            max_branching=8,
            hidden_dim=3072,
            dropout=dropout
        ).to(DEVICE)
        
        # Adapt pixel (768) to Recursive dim (1536) before recursion
        self.pixel_adapter = nn.Sequential(
            nn.Linear(768, 1536),
            nn.LayerNorm(1536),
            nn.GELU()
        ).to(DEVICE)
        
        # --- DECODER ---
        self.decoder = QwenDecoder(device=DEVICE)

    def get_trainable_parameters(self):
        params = []
        params.extend(self.pixel_embedder.parameters())
        params.extend(self.projector.parameters())
        params.extend(self.recursive_encoder.parameters())
        params.extend(self.pixel_adapter.parameters())
        return params

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.get_trainable_parameters())

    def forward(self, code: str, target: str = None, features: dict = None):
        # 1. Extract Features
        if features is None:
            features = self.extractor(code)
            
        if not features['texts']:
            if target:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return None
            
        # 2. Base Embeddings
        # [N, 768]
        cls_embeddings = self.encoder(features['texts'])
        
        depth_ids = torch.tensor(features['depths'], device=DEVICE)
        type_ids = torch.tensor(features['type_ids'], device=DEVICE)
        
        # [N, 768]
        pixel_embeddings = self.pixel_embedder(cls_embeddings, depth_ids, type_ids)
        
        # --- PATH 1: Sequential ---
        # [M, 1536]
        patch_embeddings = self.patch_embedder(pixel_embeddings)
        seq_vectors = self.projector(patch_embeddings)
        
        # --- PATH 2: Structural ---
        # Adapt pixels for RvNN: [N, 768] -> [1, N, 1536] (Batch=1 for recursion)
        pixels_for_tree = self.pixel_adapter(pixel_embeddings).unsqueeze(0)
        
        # Traverse Tree
        # [1, 1536]
        global_vector = self.recursive_encoder.forward_tree(
            features['tree_roots'], 
            pixels_for_tree
        )
        
        # --- FUSION ---
        # Concatenate: [Global(1) + Sequence(M)]
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
