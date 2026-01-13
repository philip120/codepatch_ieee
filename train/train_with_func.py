# train/train_with_func.py
"""
Semantic ViT Training - Pixel Grouping Approach

Each semantic operation = 1 pixel
Group pixels into patches (like ViT)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModel

from train.semantic_adapter import code_to_nodes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# STEP 2: CODEBERT EMBEDDINGS
# ==============================================================================

class CodeBERTEncoder:
    """
    Frozen CodeBERT encoder - extracts CLS token for each text.

    Input:  list of text strings (one per pixel)
    Output: [N, 768] tensor of CLS embeddings
    """

    def __init__(self):
        print("Loading CodeBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.to(DEVICE)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"CodeBERT loaded on {DEVICE}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Encode list of texts, return CLS token for each.

        Args:
            texts: list of N strings

        Returns:
            [N, 768] tensor
        """
        if not texts:
            return torch.zeros(0, 768, device=DEVICE)

        # Tokenize all texts
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(DEVICE)

        # Forward through CodeBERT
        outputs = self.model(**tokens)

        # Extract CLS token (first token) for each text
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [N, 768]

        return cls_embeddings

# ==============================================================================
# STEP 3: DEPTH + TYPE EMBEDDINGS
# ==============================================================================

import torch.nn as nn


class PixelEmbedder(nn.Module):
    """
    Combines CodeBERT CLS + depth embedding + type embedding.

    pixel_embedding = CLS + depth_emb + type_emb

    Trainable: depth_embedding, type_embedding
    Frozen: CodeBERT (passed in from outside)
    """

    def __init__(self, max_depth: int = 16, num_types: int = 16, embed_dim: int = 768):
        super().__init__()

        # Trainable embeddings
        self.depth_embedding = nn.Embedding(max_depth, embed_dim)
        self.type_embedding = nn.Embedding(num_types, embed_dim)

        # Initialize with small values (so they don't dominate CLS initially)
        nn.init.normal_(self.depth_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        cls_embeddings: torch.Tensor,  # [N, 768] from CodeBERT
        depth_ids: torch.Tensor,        # [N] depth indices
        type_ids: torch.Tensor,         # [N] type indices
    ) -> torch.Tensor:
        """
        Combine CLS + depth + type embeddings.

        Returns: [N, 768] pixel embeddings
        """
        depth_emb = self.depth_embedding(depth_ids)  # [N, 768]
        type_emb = self.type_embedding(type_ids)      # [N, 768]

        # Combine by addition
        pixel_embeddings = cls_embeddings + depth_emb + type_emb  # [N, 768]

        return pixel_embeddings


# ==============================================================================
# STEP 1: SEMANTIC EXTRACTION
# ==============================================================================

# Type vocabulary - maps operation types to embedding indices
TYPE_TO_ID = {
    'function': 0,
    'if': 1,
    'elseif': 2,
    'else': 3,
    'for': 4,
    'while': 5,
    'switch': 6,
    'case': 7,
    'otherwise': 8,
    'assignment': 9,
    'call': 10,
    'return': 11,
    'break': 12,
    'continue': 13,
    'try': 14,
    'catch': 15,
}
NUM_TYPES = len(TYPE_TO_ID)
MAX_DEPTH = 16


def extract_semantic_features(code: str) -> dict:
    """
    STEP 1: Extract semantic operations from MATLAB code.

    Each operation becomes a "pixel" that will later be grouped into patches.

    Args:
        code: MATLAB source code string

    Returns:
        dict with:
            - texts: list of operation text strings (for CodeBERT)
            - depths: list of depth integers (for depth embedding)
            - type_ids: list of type embedding indices (for type embedding)
    """
    ops = code_to_nodes(code, as_objects=True)

    if not ops:
        return {'texts': [], 'depths': [], 'type_ids': []}

    texts = [op.text for op in ops]
    depths = [min(op.depth, MAX_DEPTH - 1) for op in ops]
    type_ids = [TYPE_TO_ID.get(op.type, 9) for op in ops]  # default: assignment

    return {
        'texts': texts,
        'depths': depths,
        'type_ids': type_ids,
    }


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    test_code = """
    function y = test(x)
        if x > 0
            y = x * 2;
            disp('positive');
        elseif x < 0
            y = -x;
        else
            y = 0;
        end
        for i = 1:10
            disp(i);
        end
    end
    """

    print("STEP 1: SEMANTIC EXTRACTION")
    print("=" * 60)

    features = extract_semantic_features(test_code)
    ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}

    print(f"\n{len(features['texts'])} pixels extracted:\n")
    for i, (text, depth, tid) in enumerate(zip(
        features['texts'], features['depths'], features['type_ids']
    )):
        print(f"  pixel[{i}]: depth={depth}, type={ID_TO_TYPE[tid]:<10}, text={text[:35]}")

    print("\n" + "=" * 60)
    print("WHAT WILL BE FED TO CODEBERT (one per pixel):")
    print("=" * 60)
    for i, text in enumerate(features['texts']):
        print(f"  CodeBERT input[{i}]: \"{text}\"")

    print("\n" + "=" * 60)
    print("WHAT WILL BE FED TO DEPTH EMBEDDING:")
    print("=" * 60)
    print(f"  depth_ids = {features['depths']}")

    print("\n" + "=" * 60)
    print("WHAT WILL BE FED TO TYPE EMBEDDING:")
    print("=" * 60)
    print(f"  type_ids = {features['type_ids']}")
    print(f"  types    = {[ID_TO_TYPE[t] for t in features['type_ids']]}")

    print("\n" + "=" * 60)
    print("STEP 2: CODEBERT EMBEDDINGS")
    print("=" * 60)

    encoder = CodeBERTEncoder()
    cls_embeddings = encoder.encode(features['texts'])

    print(f"\n  Input:  {len(features['texts'])} texts")
    print(f"  Output: {cls_embeddings.shape}  (N pixels × 768 dim)")
    print(f"\n  Each row is a CLS embedding for one pixel:")
    for i, text in enumerate(features['texts'][:3]):  # Show first 3
        emb = cls_embeddings[i]
        print(f"    pixel[{i}]: \"{text[:30]}...\"")
        print(f"             → CLS[0:5] = {emb[:5].tolist()}")

    if len(features['texts']) > 3:
        print(f"    ... and {len(features['texts']) - 3} more pixels")

    print("\n" + "=" * 60)
    print("STEP 3: DEPTH + TYPE EMBEDDINGS")
    print("=" * 60)

    # Create embedder
    embedder = PixelEmbedder(max_depth=MAX_DEPTH, num_types=NUM_TYPES).to(DEVICE)

    # Convert to tensors
    depth_ids = torch.tensor(features['depths'], device=DEVICE)
    type_ids = torch.tensor(features['type_ids'], device=DEVICE)

    # Combine: pixel = CLS + depth_emb + type_emb
    pixel_embeddings = embedder(cls_embeddings, depth_ids, type_ids)

    print(f"\n  CLS embeddings:   {cls_embeddings.shape}")
    print(f"  + depth_emb:      {embedder.depth_embedding(depth_ids).shape}")
    print(f"  + type_emb:       {embedder.type_embedding(type_ids).shape}")
    print(f"  = pixel_emb:      {pixel_embeddings.shape}")

    print(f"\n  Trainable parameters:")
    print(f"    depth_embedding: {embedder.depth_embedding.weight.shape} = {embedder.depth_embedding.weight.numel():,} params")
    print(f"    type_embedding:  {embedder.type_embedding.weight.shape} = {embedder.type_embedding.weight.numel():,} params")

    print(f"\n  Example pixel embedding (first 5 values):")
    print(f"    pixel[0] = {pixel_embeddings[0][:5].tolist()}")

    print("\n" + "=" * 60)
    print("Ready for STEP 4: Patching (group pixels)")
    print("=" * 60)
