# train/train_with_func.py
"""
Semantic ViT Training - Main Entry Point

Uses modular components from model/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.semantic_extractor import SemanticExtractor, ID_TO_TYPE, MAX_DEPTH, NUM_TYPES
from model.codebert_encoder import CodeBERTEncoder
from model.pixel_embedder import PixelEmbedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# TEST ALL STEPS
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

    # ===================== STEP 1 =====================
    print("=" * 60)
    print("STEP 1: SEMANTIC EXTRACTION")
    print("=" * 60)

    extractor = SemanticExtractor()
    features = extractor(test_code)

    print(f"\n{len(features['texts'])} pixels extracted:\n")
    for i, (text, depth, tid) in enumerate(zip(
        features['texts'], features['depths'], features['type_ids']
    )):
        print(f"  pixel[{i}]: depth={depth}, type={ID_TO_TYPE[tid]:<10}, text=\"{text[:35]}\"")

    print("\n  CodeBERT will receive:")
    for i, text in enumerate(features['texts']):
        print(f"    [{i}] \"{text}\"")

    # ===================== STEP 2 =====================
    print("\n" + "=" * 60)
    print("STEP 2: CODEBERT EMBEDDINGS")
    print("=" * 60)

    encoder = CodeBERTEncoder(device=DEVICE)
    cls_embeddings = encoder(features['texts'])

    print(f"\n  Input:  {len(features['texts'])} texts")
    print(f"  Output: {cls_embeddings.shape}  (N x 768)")

    # ===================== STEP 3 =====================
    print("\n" + "=" * 60)
    print("STEP 3: PIXEL EMBEDDINGS (CLS + depth + type)")
    print("=" * 60)

    embedder = PixelEmbedder(max_depth=MAX_DEPTH, num_types=NUM_TYPES).to(DEVICE)

    depth_ids = torch.tensor(features['depths'], device=DEVICE)
    type_ids = torch.tensor(features['type_ids'], device=DEVICE)

    pixel_embeddings = embedder(cls_embeddings, depth_ids, type_ids)

    print(f"\n  CLS [N, 768]:        {cls_embeddings.shape}")
    print(f"  + depth_emb [N, 768]")
    print(f"  + type_emb [N, 768]")
    print(f"  = pixel_emb [N, 768]: {pixel_embeddings.shape}")
    print(f"\n  Trainable params: {embedder.num_parameters():,}")

    # ===================== SUMMARY =====================
    print("\n" + "=" * 60)
    print("SUMMARY SO FAR")
    print("=" * 60)
    print(f"""
  MATLAB code
      │
      ▼ SemanticExtractor (Step 1)
  {len(features['texts'])} pixels: [texts, depths, type_ids]
      │
      ▼ CodeBERTEncoder (Step 2, frozen)
  cls_embeddings: {cls_embeddings.shape}
      │
      ▼ PixelEmbedder (Step 3, trainable)
  pixel_embeddings: {pixel_embeddings.shape}
      │
      ▼ PatchEmbedder (Step 4, TODO)
  patch_embeddings: [N/P, P*768]
      │
      ▼ Projector (Step 5, TODO)
  projected: [N/P, 1536]
      │
      ▼ Qwen (Step 6, TODO)
  output text
    """)

    print("=" * 60)
    print("Ready for STEP 4: Patching")
    print("=" * 60)
