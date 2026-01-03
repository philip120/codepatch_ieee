# train/inference_pure.py
"""
Inference for Pure AST-to-Thought model.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.train_pure import PureASTModel, DEVICE
from train.semantic_adapter import code_to_nodes


def load_model(checkpoint_path: str = "pure_best.pt", max_nodes: int = 32):
    """Load trained model."""
    print("Loading model...")
    model = PureASTModel(max_nodes=max_nodes).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.projector.load_state_dict(checkpoint['projector_state_dict'])
    model.eval()

    return model


def generate(
    matlab_code: str,
    model: PureASTModel,
    max_new_tokens: int = 128,
    show_nodes: bool = True,
) -> str:
    """Generate pseudocode from MATLAB code."""
    nodes = code_to_nodes(matlab_code)

    if not nodes:
        return "Error: No nodes extracted"

    if show_nodes:
        print(f"\nExtracted {len(nodes)} AST nodes:")
        for i, n in enumerate(nodes):
            print(f"  [{i}] {n}")

    output = model.generate(nodes, max_new_tokens=max_new_tokens)
    return output


def interactive(checkpoint_path: str = "pure_best.pt"):
    """Interactive mode."""
    print("=" * 60)
    print("Pure AST-to-Thought Pseudocode Generator")
    print("=" * 60)
    print("\nEach AST node → CodeBERT → MLP → Qwen token")
    print("Qwen sees ALL tokens at once, then generates.\n")

    model = load_model(checkpoint_path)

    while True:
        print("-" * 40)
        print("Enter MATLAB code (empty line to generate, 'quit' to exit):")

        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.lower() == "quit":
                return
            if line == "":
                break
            lines.append(line)

        if not lines:
            continue

        code = "\n".join(lines)
        print("\nGenerating...")

        output = generate(code, model)

        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pure_best.pt")
    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    if args.code:
        model = load_model(args.checkpoint)
        out = generate(args.code, model, args.max_tokens)
        print("\nPSEUDOCODE:\n" + out)
    elif args.file:
        with open(args.file) as f:
            code = f.read()
        model = load_model(args.checkpoint)
        out = generate(code, model, args.max_tokens)
        print("\nPSEUDOCODE:\n" + out)
    else:
        interactive(args.checkpoint)
