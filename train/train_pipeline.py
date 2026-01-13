# train/train_pipeline.py
"""
Full Training Pipeline for Semantic ViT

Usage:
    python -m train.train_pipeline --epochs 10 --batch_size 1 --lr 1e-4
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path

from model.semantic_extractor import SemanticExtractor, MAX_DEPTH, NUM_TYPES
from model.codebert_encoder import CodeBERTEncoder
from model.pixel_embedder import PixelEmbedder
from model.patch_embedder import PatchEmbedder
from model.projector import Projector
from model.qwen_decoder import QwenDecoder

from train.load_dataset import load_matlab_nl_dataset
from train.matlab_dataset import MatlabPseudocodeDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ==============================================================================
# DATASET
# ==============================================================================

# Dataset class moved to train/matlab_dataset.py


# ==============================================================================
# MODEL WRAPPER
# ==============================================================================

class SemanticViT(nn.Module):
    """
    Full Semantic ViT model wrapping all components.

    Trainable: PixelEmbedder, Projector
    Frozen: CodeBERTEncoder, QwenDecoder
    """

    def __init__(
        self,
        patch_size: int = 4,
        bottleneck_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.patch_size = patch_size

        # Step 1: Semantic extraction (not nn.Module)
        self.extractor = SemanticExtractor()

        # Step 2: CodeBERT (frozen, not nn.Module)
        self.encoder = CodeBERTEncoder(device=DEVICE)

        # Step 3: Pixel embedder (TRAINABLE)
        self.pixel_embedder = PixelEmbedder(
            max_depth=MAX_DEPTH,
            num_types=NUM_TYPES,
        ).to(DEVICE)

        # Step 4: Patch embedder (no params)
        self.patch_embedder = PatchEmbedder(patch_size=patch_size)

        # Step 5: Projector (TRAINABLE)
        self.projector = Projector(
            in_dim=patch_size * 768,
            bottleneck_dim=bottleneck_dim,
            out_dim=1536,
            dropout=dropout,
        ).to(DEVICE)

        # Step 6: Qwen decoder (frozen, not nn.Module)
        self.decoder = QwenDecoder(device=DEVICE)

    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        params = []
        params.extend(self.pixel_embedder.parameters())
        params.extend(self.projector.parameters())
        return params

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.get_trainable_parameters())

    def forward(self, code: str, target: str = None, features: dict = None):
        """
        Forward pass.

        Args:
            code: MATLAB source code
            target: Target pseudocode (for training)
            features: Pre-computed semantic features (optional)

        Returns:
            If target provided: loss
            If no target: projected embeddings
        """
        # Step 1: Extract semantic features
        if features is None:
            features = self.extractor(code)

        if not features['texts']:
            # Empty code, return dummy loss
            if target:
                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
            return None

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

        # Step 6: Decoder
        if target:
            # Training: compute loss
            loss = self.decoder.forward_train(projected, target)
            return loss
        else:
            # Inference: return projected for generation
            return projected

    @torch.no_grad()
    def generate(self, code: str, max_new_tokens: int = 128) -> str:
        """Generate pseudocode from MATLAB code."""
        projected = self.forward(code, target=None)
        if projected is None:
            return ""
        return self.decoder.generate(projected, max_new_tokens=max_new_tokens)


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(
    split: str = "train",
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    patch_size: int = 4,
    bottleneck_dim: int = 512,
    dropout: float = 0.4,
    log_every: int = 10,
    eval_every: int = 50,
    save_every: int = 100,
    save_dir: str = "checkpoints",
    gradient_accumulation: int = 4,
):
    """
    Main training function.
    """
    print("=" * 60)
    print("SEMANTIC VIT TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patch size: {patch_size}")
    print(f"Bottleneck: {bottleneck_dim}")
    print(f"Dropout: {dropout}")
    print(f"Gradient accumulation: {gradient_accumulation}")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Load dataset from Hugging Face
    print("\n" + "=" * 60)
    print("Loading dataset from Hugging Face...")
    dataset = MatlabPseudocodeDataset(split=split)
    # Use collate_fn to return single sample dict directly (batch_size=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    model = SemanticViT(
        patch_size=patch_size,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
    )

    print(f"\nTrainable parameters: {model.num_trainable_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    total_steps = epochs * len(loader) // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    global_step = 0
    accumulation_step = 0
    running_loss = 0.0
    best_loss = float('inf')

    model.pixel_embedder.train()
    model.projector.train()

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, batch in enumerate(loader):
            code = batch['code']
            target = batch['target']
            features = batch.get('features')

            # Forward pass
            loss = model(code, target=target, features=features)

            if loss is None or loss.item() == 0:
                continue

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation
            loss.backward()

            accumulation_step += 1
            running_loss += loss.item() * gradient_accumulation
            epoch_loss += loss.item() * gradient_accumulation
            epoch_samples += 1

            # Update weights after accumulation
            if accumulation_step % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % log_every == 0:
                    avg_loss = running_loss / (log_every * gradient_accumulation)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  step {global_step:4d} | loss {avg_loss:.4f} | lr {lr_now:.2e}")
                    running_loss = 0.0

                # Evaluation
                if global_step % eval_every == 0:
                    model.pixel_embedder.eval()
                    model.projector.eval()

                    print(f"\n  --- Eval at step {global_step} ---")
                    print(f"  Code: {code[:80]}...")
                    print(f"  Target: {target[:80]}...")

                    generated = model.generate(code, max_new_tokens=64)
                    print(f"  Generated: {generated[:80]}...")
                    print()

                    model.pixel_embedder.train()
                    model.projector.train()

                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint = {
                        'step': global_step,
                        'epoch': epoch,
                        'pixel_embedder': model.pixel_embedder.state_dict(),
                        'projector': model.projector.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': epoch_loss / max(epoch_samples, 1),
                    }
                    torch.save(checkpoint, save_path / f"checkpoint_{global_step}.pt")
                    print(f"  Saved checkpoint_{global_step}.pt")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        print(f"\nEpoch {epoch + 1} complete | avg_loss: {avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint = {
                'step': global_step,
                'epoch': epoch,
                'pixel_embedder': model.pixel_embedder.state_dict(),
                'projector': model.projector.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, save_path / "best_model.pt")
            print(f"  New best model! Loss: {best_loss:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_path}")
    print("=" * 60)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Semantic ViT")

    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (train/test)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--grad_accum", type=int, default=4)

    args = parser.parse_args()

    train(
        split=args.split,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        bottleneck_dim=args.bottleneck,
        dropout=args.dropout,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        gradient_accumulation=args.grad_accum,
    )
