# train/inference_e2e.py
"""
Inference for End-to-End trained model.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.train_e2e import CodeToThoughtModel, DEVICE
from train.semantic_adapter import code_to_nodes
from train.model import codebert_embed_nodes


def load_e2e_model(checkpoint_path: str = "e2e_best.pt", num_thoughts: int = 8):
    """Load trained end-to-end model."""
    print("Loading model...")
    model = CodeToThoughtModel(num_thought_tokens=num_thoughts).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.mlp.load_state_dict(checkpoint['mlp_state_dict'])
    model.thought_expander.load_state_dict(checkpoint['expander_state_dict'])
    model.eval()

    return model


def generate_pseudocode(
    matlab_code: str,
    model: CodeToThoughtModel,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    show_nodes: bool = True,
) -> str:
    """Generate pseudocode from MATLAB code."""

    # Extract semantic nodes
    nodes = code_to_nodes(matlab_code)
    if not nodes:
        return "Error: Could not parse MATLAB code"

    if show_nodes:
        print(f"\nExtracted {len(nodes)} semantic nodes:")
        for i, n in enumerate(nodes):
            print(f"  [{i}] {n}")

    # Get CodeBERT embedding
    with torch.no_grad():
        code_emb = codebert_embed_nodes(nodes)

        # Generate
        output = model.generate(
            code_emb,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return output


def interactive_mode(checkpoint_path: str = "e2e_best.pt", num_thoughts: int = 8):
    """Interactive pseudocode generation."""
    print("=" * 60)
    print("MATLAB to Pseudocode (End-to-End Model)")
    print("=" * 60)

    model = load_e2e_model(checkpoint_path, num_thoughts)

    while True:
        print("\n" + "-" * 40)
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

        matlab_code = "\n".join(lines)

        print("\nGenerating...")
        output = generate_pseudocode(matlab_code, model)

        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="e2e_best.pt")
    parser.add_argument("--thoughts", type=int, default=8)
    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.code:
        model = load_e2e_model(args.checkpoint, args.thoughts)
        output = generate_pseudocode(args.code, model, args.max_tokens, args.temperature)
        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(output)
    elif args.file:
        with open(args.file) as f:
            code = f.read()
        model = load_e2e_model(args.checkpoint, args.thoughts)
        output = generate_pseudocode(code, model, args.max_tokens, args.temperature)
        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(output)
    else:
        interactive_mode(args.checkpoint, args.thoughts)
