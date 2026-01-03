# train/train_loop.py
import sys
import os

# Add project root to path for imports when running from terminal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.load_dataset import load_matlab_nl_dataset
from train.semantic_adapter import code_to_nodes
from train.dataset import CodeNLDataset

from train.model import (
    codebert_embed_nodes,
    qwen_embed_text,
    ProjectionMLP,
    cosine_loss,
    combined_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_loss(z_pred, z_target, loss_type="combined"):
    """Compute loss based on type."""
    if loss_type == "mse":
        return F.mse_loss(z_pred, z_target)
    elif loss_type == "cosine":
        return cosine_loss(z_pred, z_target)
    elif loss_type == "combined":
        return combined_loss(z_pred, z_target, alpha=0.7)  # 70% cosine, 30% MSE
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main(
    num_epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-4,
    save_every: int = 200,
    log_every: int = 20,
    max_steps: int = None,
    loss_type: str = "combined",
):
    print(f"Device: {DEVICE}")
    print(f"Loss type: {loss_type}")
    print("Loading dataset from HuggingFace...")

    raw = load_matlab_nl_dataset("train")
    print(f"Loaded {len(raw)} raw examples")

    dataset = CodeNLDataset(raw, code_to_nodes)
    print(f"Dataset size after filtering: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    mlp = ProjectionMLP(
        in_dim=768,
        out_dim=1536
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(loader), eta_min=1e-6
    )

    global_step = 0
    running_loss = 0.0
    running_cos_sim = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        epoch_cos_sim = 0.0
        epoch_steps = 0

        for step, batch in enumerate(loader):
            code_nodes = batch["code_nodes"][0]
            nl_text = batch["nl_text"][0]

            z_code = codebert_embed_nodes(code_nodes)   # [768]
            z_target = qwen_embed_text(nl_text)         # [1536]

            z_pred = mlp(z_code)
            loss = compute_loss(z_pred, z_target, loss_type)

            # Track cosine similarity for monitoring
            with torch.no_grad():
                cos_sim = F.cosine_similarity(z_pred.unsqueeze(0), z_target.unsqueeze(0)).item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_cos_sim += cos_sim
            epoch_loss += loss.item()
            epoch_cos_sim += cos_sim
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                avg_cos = running_cos_sim / log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {global_step} | loss {avg_loss:.4f} | cos_sim {avg_cos:.4f} | lr {lr_now:.2e}")
                running_loss = 0.0
                running_cos_sim = 0.0

            if save_every and global_step % save_every == 0:
                save_path = f"mlp_checkpoint_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'cos_sim': cos_sim,
                }, save_path)
                print(f"Saved checkpoint to {save_path}")

            if max_steps and global_step >= max_steps:
                print(f"Reached max_steps ({max_steps}), stopping.")
                break

        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_steps
        avg_epoch_cos = epoch_cos_sim / epoch_steps
        print(f"Epoch {epoch+1} complete | avg_loss {avg_epoch_loss:.4f} | avg_cos_sim {avg_epoch_cos:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'model_state_dict': mlp.state_dict(),
                'loss': avg_epoch_loss,
                'cos_sim': avg_epoch_cos,
            }, "mlp_best.pt")
            print(f"New best model saved! (loss: {best_loss:.4f})")

        if max_steps and global_step >= max_steps:
            break

    # Final save
    final_path = "mlp_final.pt"
    torch.save({
        'step': global_step,
        'model_state_dict': mlp.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best model saved to mlp_best.pt (loss: {best_loss:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--loss", type=str, default="combined", choices=["mse", "cosine", "combined"])
    args = parser.parse_args()

    main(
        num_epochs=args.epochs,
        lr=args.lr,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        loss_type=args.loss,
    )
