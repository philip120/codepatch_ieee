# train/evaluate.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from train.load_dataset import load_matlab_nl_dataset
from train.semantic_adapter import code_to_nodes
from train.model import (
    codebert_embed_nodes,
    qwen_embed_text,
    ProjectionMLP,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str) -> ProjectionMLP:
    """Load trained MLP from checkpoint."""
    mlp = ProjectionMLP(in_dim=768, out_dim=1536).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval()
    return mlp


def evaluate_single(mlp, code: str, nl_text: str):
    """Evaluate a single code-NL pair."""
    nodes = code_to_nodes(code)
    if not nodes:
        return None

    with torch.no_grad():
        z_code = codebert_embed_nodes(nodes)
        z_target = qwen_embed_text(nl_text)
        z_pred = mlp(z_code)

        cos_sim = F.cosine_similarity(z_pred.unsqueeze(0), z_target.unsqueeze(0)).item()
        mse = F.mse_loss(z_pred, z_target).item()

    return {
        "cos_sim": cos_sim,
        "mse": mse,
        "nodes": nodes,
    }


def evaluate_dataset(checkpoint_path: str, split: str = "train", max_samples: int = 100):
    """Evaluate model on dataset split."""
    print(f"Loading model from {checkpoint_path}...")
    mlp = load_model(checkpoint_path)

    print(f"Loading {split} split...")
    raw = load_matlab_nl_dataset(split)
    samples = raw[:max_samples]

    print(f"Evaluating {len(samples)} samples...")

    results = []
    for i, ex in enumerate(samples):
        result = evaluate_single(mlp, ex["code"], ex["nl"])
        if result:
            results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    # Aggregate metrics
    cos_sims = [r["cos_sim"] for r in results]
    mses = [r["mse"] for r in results]

    print(f"\n=== Evaluation Results ({len(results)} valid samples) ===")
    print(f"Cosine Similarity: mean={sum(cos_sims)/len(cos_sims):.4f}, "
          f"min={min(cos_sims):.4f}, max={max(cos_sims):.4f}")
    print(f"MSE: mean={sum(mses)/len(mses):.4f}, "
          f"min={min(mses):.4f}, max={max(mses):.4f}")

    return results


def interactive_test(checkpoint_path: str):
    """Interactive testing of the model."""
    print(f"Loading model from {checkpoint_path}...")
    mlp = load_model(checkpoint_path)

    print("\nInteractive mode. Enter MATLAB code (or 'quit' to exit):")
    print("=" * 50)

    while True:
        print("\nEnter MATLAB code (multi-line, end with empty line):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            if line.lower() == "quit":
                return
            lines.append(line)

        if not lines:
            continue

        code = "\n".join(lines)
        nodes = code_to_nodes(code)

        print(f"\nExtracted {len(nodes)} semantic nodes:")
        for i, n in enumerate(nodes):
            print(f"  [{i}] {n}")

        print("\nEnter expected NL description:")
        nl_text = input()

        if nl_text:
            result = evaluate_single(mlp, code, nl_text)
            if result:
                print(f"\nCosine Similarity: {result['cos_sim']:.4f}")
                print(f"MSE: {result['mse']:.4f}")

                if result['cos_sim'] > 0.8:
                    print("Excellent alignment!")
                elif result['cos_sim'] > 0.5:
                    print("Good alignment.")
                elif result['cos_sim'] > 0.2:
                    print("Moderate alignment.")
                else:
                    print("Poor alignment.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="mlp_best.pt")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "interactive"])
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_dataset(args.checkpoint, max_samples=args.max_samples)
    else:
        interactive_test(args.checkpoint)
