"""
debug_encoder.py — sanity-check the encoder pipeline

Checks:
  1. Are projected embedding norms compatible with Qwen token embedding norms?
  2. Are different code samples producing different projected outputs (cos-sim)?
  3. Is depth/type embedding contributing anything vs plain CLS passthrough?

Run:
    python debug_encoder.py --checkpoint checkpoints_stage2/best_model.pt
    python debug_encoder.py  # no checkpoint — uses random weights
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES = [
    ("function y = double_if_pos(x)\n  if x > 0\n    y = 2*x;\n  else\n    y = 0;\n  end\nend",
     "if-else branch"),
    ("for i = 1:10\n  s = s + i;\nend",
     "for loop"),
    ("A = zeros(n, n);\nfor i = 1:n\n  A(i,i) = 1;\nend",
     "identity matrix"),
    ("function out = relu(x)\n  out = max(0, x);\nend",
     "relu"),
    ("x = input('Enter: ');\ny = x^2;\ndisp(y)",
     "read and square"),
]


def load_model(checkpoint_path=None, patch_size=4, bottleneck_dim=768, dropout=0.05):
    from combined_model.model import CombinedSemanticViT
    model = CombinedSemanticViT(patch_size=patch_size, bottleneck_dim=bottleneck_dim, dropout=dropout)

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        if "model_state" in ckpt:
            mp = dict(model.named_parameters())
            loaded = sum(1 for n, d in ckpt["model_state"].items()
                         if n in mp and not mp[n].data.copy_(d) is None)
            print(f"  Loaded {len(ckpt['model_state'])} encoder tensors")

        if "qwen_state" in ckpt and ckpt["qwen_state"]:
            idxs = set()
            for k in ckpt["qwen_state"]:
                parts = k.split(".")
                if len(parts) > 2 and parts[1] == "layers":
                    idxs.add(int(parts[2]))
            model.decoder.unfreeze_layers(len(idxs))
            model.decoder.load_unfrozen_state_dict(ckpt["qwen_state"])
            print(f"  Loaded {len(ckpt['qwen_state'])} unfrozen Qwen tensors")
    else:
        print("No checkpoint — using random encoder weights")

    model.eval()
    return model


@torch.no_grad()
def run_debug(model):
    print("\n" + "=" * 60)
    print("1. NORM CHECK: projected vs Qwen token embeddings")
    print("=" * 60)

    # Qwen token embedding norms
    qwen_emb = model.decoder.model.get_input_embeddings()
    token_norms = qwen_emb.weight.norm(dim=-1)
    print(f"  Qwen token embedding norms:")
    print(f"    mean = {token_norms.mean().item():.4f}")
    print(f"    std  = {token_norms.std().item():.4f}")
    print(f"    min  = {token_norms.min().item():.4f}")
    print(f"    max  = {token_norms.max().item():.4f}")

    # Projected embedding norms for each sample
    print(f"\n  Projected embedding norms (per sample):")
    projected_list = []
    for code, label in SAMPLES:
        proj = model.forward(code)
        if proj is None:
            print(f"    [{label}] → EMPTY (extractor returned nothing)")
            continue
        norms = proj.norm(dim=-1)
        print(f"    [{label}] patches={proj.shape[0]}  "
              f"norm_mean={norms.mean().item():.3f}  "
              f"norm_std={norms.std().item():.3f}  "
              f"var={proj.var(dim=0).mean().item():.4f}")
        projected_list.append((label, proj))

    print("\n" + "=" * 60)
    print("2. DISCRIMINATION: cosine similarity between different codes")
    print("   (should be < 0.9 if encoder is distinguishing samples)")
    print("=" * 60)

    if len(projected_list) >= 2:
        for i in range(len(projected_list)):
            for j in range(i + 1, len(projected_list)):
                la, pa = projected_list[i]
                lb, pb = projected_list[j]
                # Compare mean projected vectors
                va = pa.mean(dim=0)
                vb = pb.mean(dim=0)
                cos = torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
                print(f"  [{la}] vs [{lb}]: cos_sim = {cos:.4f}")

    print("\n" + "=" * 60)
    print("3. DEPTH/TYPE CONTRIBUTION")
    print("   CLS alone vs CLS + depth + type")
    print("=" * 60)

    # Test on first sample
    code, label = SAMPLES[0]
    feats = model.extractor(code)
    if feats['texts']:
        cls = model.encoder(feats['texts'])
        depth_ids = torch.tensor(feats['depths'], device=DEVICE)
        type_ids = torch.tensor(feats['type_ids'], device=DEVICE)

        pixel = model.pixel_embedder(cls, depth_ids, type_ids)

        # How much do depth/type add relative to CLS?
        depth_emb = model.pixel_embedder.depth_embedding(depth_ids)
        type_emb = model.pixel_embedder.type_embedding(type_ids)

        cls_norm = cls.norm(dim=-1).mean().item()
        depth_norm = depth_emb.norm(dim=-1).mean().item()
        type_norm = type_emb.norm(dim=-1).mean().item()
        pixel_norm = pixel.norm(dim=-1).mean().item()

        cos_cls_pixel = torch.nn.functional.cosine_similarity(
            cls.mean(dim=0, keepdim=True),
            pixel.mean(dim=0, keepdim=True)
        ).item()

        print(f"  Sample: [{label}]  N={cls.shape[0]} nodes")
        print(f"  CLS norm:   {cls_norm:.4f}")
        print(f"  depth norm: {depth_norm:.4f}  ({100*depth_norm/cls_norm:.1f}% of CLS)")
        print(f"  type norm:  {type_norm:.4f}  ({100*type_norm/cls_norm:.1f}% of CLS)")
        print(f"  pixel norm: {pixel_norm:.4f}")
        print(f"  cos_sim(CLS_mean, pixel_mean) = {cos_cls_pixel:.4f}")
        print(f"  → depth/type are {'negligible' if depth_norm/cls_norm < 0.1 else 'significant'}")

    print("\n" + "=" * 60)
    print("4. GENERATION SANITY CHECK")
    print("=" * 60)
    for code, label in SAMPLES[:3]:
        out = model.generate(code, max_new_tokens=64)
        print(f"\n  [{label}]")
        print(f"  Code: {code.strip()[:80]}")
        print(f"  Out:  {out[:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.patch_size, args.bottleneck, args.dropout)
    run_debug(model)
