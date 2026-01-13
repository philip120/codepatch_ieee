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
from model.patch_embedder import PatchEmbedder
from model.projector import Projector
from model.qwen_decoder import QwenDecoder

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

    # ===================== STEP 4 =====================
    print("\n" + "=" * 60)
    print("STEP 4: PATCH EMBEDDINGS (group pixels)")
    print("=" * 60)

    PATCH_SIZE = 4
    patcher = PatchEmbedder(patch_size=PATCH_SIZE)

    patch_embeddings = patcher(pixel_embeddings)

    num_pixels = pixel_embeddings.shape[0]
    num_patches = patch_embeddings.shape[0]
    flat_dim = patch_embeddings.shape[1]

    print(f"\n  patch_size = {PATCH_SIZE}")
    print(f"\n  Input:  {num_pixels} pixels  [{num_pixels}, 768]")
    print(f"  Output: {num_patches} patches [{num_patches}, {flat_dim}]")
    print(f"\n  Visualization:")
    print(f"    pixels:  [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, PAD, PAD]")
    print(f"              \_________/  \_________/  \______________/")
    print(f"    patches:   patch 0      patch 1        patch 2")
    print(f"\n  Each patch: {PATCH_SIZE} pixels × 768 = {flat_dim} dim")

    # ===================== STEP 5 =====================
    print("\n" + "=" * 60)
    print("STEP 5: PROJECTOR (Bottleneck MLP)")
    print("=" * 60)

    BOTTLENECK = 512
    QWEN_DIM = 1536
    DROPOUT = 0.4

    projector = Projector(
        in_dim=flat_dim,
        bottleneck_dim=BOTTLENECK,
        out_dim=QWEN_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    projected = projector(patch_embeddings)

    print(f"\n  Architecture: {flat_dim} → {BOTTLENECK} → {QWEN_DIM}")
    print(f"  Dropout: {DROPOUT}")
    print(f"\n  Input:  {patch_embeddings.shape}")
    print(f"  Output: {projected.shape}")
    print(f"\n  Trainable params: {projector.num_parameters():,}")

    # ===================== SUMMARY =====================
    print("\n" + "=" * 60)
    print("SUMMARY - FULL PIPELINE")
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
      ▼ PixelEmbedder (Step 3, trainable: {embedder.num_parameters():,} params)
  pixel_embeddings: {pixel_embeddings.shape}
      │
      ▼ PatchEmbedder (Step 4, patch_size={PATCH_SIZE})
  patch_embeddings: {patch_embeddings.shape}
      │
      ▼ Projector (Step 5, trainable: {projector.num_parameters():,} params)
  projected: {projected.shape}
      │
      ▼ Qwen (Step 6, frozen, TODO)
  output text

  TOTAL TRAINABLE: {embedder.num_parameters() + projector.num_parameters():,} params
    """)

    # ===================== STEP 6 =====================
    print("=" * 60)
    print("STEP 6: QWEN DECODER")
    print("=" * 60)

    decoder = QwenDecoder(device=DEVICE)

    # Test training forward pass
    target_text = "This function tests if x is positive and doubles it, otherwise sets y to zero."

    print(f"\n  Target text: \"{target_text[:50]}...\"")
    print(f"  Projected shape: {projected.shape}")

    loss = decoder.forward_train(projected, target_text)
    print(f"\n  Training loss: {loss.item():.4f}")

    # Test generation
    print("\n  Generating from embeddings...")
    generated = decoder.generate(projected, max_new_tokens=32)
    print(f"  Generated: \"{generated[:100]}...\"")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE")
    print("=" * 60)
    print(f"""
  MATLAB code
      │
      ▼ SemanticExtractor (Step 1)
  {len(features['texts'])} pixels
      │
      ▼ CodeBERTEncoder (Step 2, FROZEN)
  {cls_embeddings.shape}
      │
      ▼ PixelEmbedder (Step 3, TRAINABLE: {embedder.num_parameters():,})
  {pixel_embeddings.shape}
      │
      ▼ PatchEmbedder (Step 4, patch_size={PATCH_SIZE})
  {patch_embeddings.shape}
      │
      ▼ Projector (Step 5, TRAINABLE: {projector.num_parameters():,})
  {projected.shape}
      │
      ▼ QwenDecoder (Step 6, FROZEN)
  → loss: {loss.item():.4f}
  → text: "{generated[:50]}..."

  ════════════════════════════════════════════════════════
  TRAINABLE PARAMS: {embedder.num_parameters() + projector.num_parameters():,}
  FROZEN PARAMS:    CodeBERT (~125M) + Qwen (~1.5B)
  ════════════════════════════════════════════════════════
    """)

    print("=" * 60)
    print("ALL STEPS COMPLETE! Ready for training loop.")
    print("=" * 60)
