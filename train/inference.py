# train/inference.py
"""
Semantic ViT Inference Pipeline

Usage:
    python -m train.inference --checkpoint checkpoints/best_model.pt --eval
    python -m train.inference --checkpoint checkpoints/best_model.pt --code "function y = test(x) ..."
    python -m train.inference --checkpoint checkpoints/best_model.pt --interactive
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path

from model.semantic_extractor import SemanticExtractor, MAX_DEPTH, NUM_TYPES
from model.codebert_encoder import CodeBERTEncoder
from model.pixel_embedder import PixelEmbedder
from model.patch_embedder import PatchEmbedder
from model.projector import Projector
from model.qwen_decoder import QwenDecoder

from train.load_dataset import load_matlab_nl_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticViTInference:
    """
    Inference wrapper for Semantic ViT.
    """

    def __init__(
        self,
        checkpoint_path: str,
        patch_size: int = 4,
        bottleneck_dim: int = 512,
    ):
        print("=" * 60)
        print("Loading Semantic ViT for Inference")
        print("=" * 60)

        self.patch_size = patch_size

        # Step 1: Semantic extraction
        self.extractor = SemanticExtractor()

        # Step 2: CodeBERT (frozen)
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # Step 3: Pixel embedder
        self.pixel_embedder = PixelEmbedder(
            max_depth=MAX_DEPTH,
            num_types=NUM_TYPES,
        ).to(DEVICE)

        # Step 4: Patch embedder
        self.patch_embedder = PatchEmbedder(patch_size=patch_size)

        # Step 5: Qwen decoder (created before projector to read hidden_size)
        self.decoder = QwenDecoder(device=DEVICE)
        qwen_dim = self.decoder.hidden_size

        # Step 6: Projector
        self.projector = Projector(
            in_dim=patch_size * 768,
            bottleneck_dim=bottleneck_dim,
            out_dim=qwen_dim,
            dropout=0.0,  # No dropout during inference
        ).to(DEVICE)

        # Load checkpoint
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        self.pixel_embedder.load_state_dict(checkpoint['pixel_embedder'])

        # Handle projector with dropout mismatch
        projector_state = checkpoint['projector']
        self.projector.load_state_dict(projector_state, strict=False)

        # Set to eval mode
        self.pixel_embedder.eval()
        self.projector.eval()

        step = checkpoint.get('step', 'N/A')
        loss = checkpoint.get('loss', 0)
        print(f"Checkpoint loaded (step {step}, loss {loss:.4f})")
        print("=" * 60)

    @torch.no_grad()
    def generate(self, code: str, max_new_tokens: int = 128, show_features: bool = False) -> str:
        """
        Generate pseudocode from MATLAB code.
        """
        # Step 1: Extract semantic features
        features = self.extractor(code)

        if not features['texts']:
            return "[Empty code - no semantic operations found]"

        if show_features:
            print(f"\n  Extracted {len(features['texts'])} semantic operations:")
            for i, (text, depth) in enumerate(zip(features['texts'], features['depths'])):
                print(f"    [{i}] D{depth}: {text[:50]}")

        # Step 2: CodeBERT embeddings
        cls_embeddings = self.encoder(features['texts'])

        # Step 3: Pixel embeddings
        depth_ids = torch.tensor(features['depths'], device=DEVICE)
        type_ids = torch.tensor(features['type_ids'], device=DEVICE)
        pixel_embeddings = self.pixel_embedder(cls_embeddings, depth_ids, type_ids)

        # Step 4: Patch embeddings
        patch_embeddings = self.patch_embedder(pixel_embeddings)

        # Step 5: Project to Qwen space
        projected = self.projector(patch_embeddings)

        # Step 6: Generate
        return self.decoder.generate(projected, max_new_tokens=max_new_tokens)


def evaluate_on_dataset(model: SemanticViTInference, split: str = "train", num_samples: int = 10):
    """
    Evaluate model on dataset samples.
    """
    print("\n" + "=" * 60)
    print(f"Evaluating on {num_samples} samples from '{split}' split")
    print("=" * 60)

    data = load_matlab_nl_dataset(split)

    for i, sample in enumerate(data[:num_samples]):
        code = sample['code']
        target = sample['nl']

        print(f"\n{'─' * 60}")
        print(f"Sample {i + 1}/{num_samples}")
        print(f"{'─' * 60}")

        # Truncate code for display
        code_display = code[:300] + "..." if len(code) > 300 else code
        print(f"\nCODE:\n{code_display}")

        print(f"\nTARGET:\n{target}")

        generated = model.generate(code, max_new_tokens=100)
        print(f"\nGENERATED:\n{generated}")

    print("\n" + "=" * 60)
    print("Evaluation complete")
    print("=" * 60)


def interactive_mode(model: SemanticViTInference):
    """
    Interactive mode - enter code and get pseudocode.
    """
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("Enter MATLAB code (type 'END' on new line to finish, 'quit' to exit)")
    print("=" * 60)

    while True:
        print("\nEnter MATLAB code:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip().upper() == 'END':
                break
            if line.strip().lower() == 'quit':
                print("Goodbye!")
                return
            lines.append(line)

        code = '\n'.join(lines)

        if not code.strip():
            continue

        print("\nGenerating...")
        generated = model.generate(code, show_features=True)
        print(f"\n{'=' * 40}")
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic ViT Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--code", type=str, default=None,
                        help="MATLAB code to process (inline)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to MATLAB file to process")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate on dataset samples")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split for evaluation")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples for evaluation")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=128)

    args = parser.parse_args()

    # Load model
    model = SemanticViTInference(
        checkpoint_path=args.checkpoint,
        patch_size=args.patch_size,
        bottleneck_dim=args.bottleneck,
    )

    # Run mode
    if args.eval:
        evaluate_on_dataset(model, split=args.split, num_samples=args.num_samples)

    elif args.code:
        print("\nInput code:")
        print(args.code)
        generated = model.generate(args.code, max_new_tokens=args.max_tokens, show_features=True)
        print(f"\n{'=' * 40}")
        print("GENERATED:")
        print("=" * 40)
        print(generated)

    elif args.file:
        code = Path(args.file).read_text()
        print(f"\nInput file: {args.file}")
        generated = model.generate(code, max_new_tokens=args.max_tokens, show_features=True)
        print(f"\n{'=' * 40}")
        print("GENERATED:")
        print("=" * 40)
        print(generated)

    elif args.interactive:
        interactive_mode(model)

    else:
        # Demo
        demo_code = """
function y = calc(x)
    y = 0;
    for i = 1:length(x)
        y = y + x(i);
    end
end
"""
        print("Demo: Generating pseudocode for sample function")
        print(f"\nCode:\n{demo_code}")
        generated = model.generate(demo_code, show_features=True)
        print(f"\n{'=' * 40}")
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(generated)
