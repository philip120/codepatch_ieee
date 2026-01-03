# train/train_loop.py
import sys
import os

# Add project root to path for imports when running from terminal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from train.load_dataset import load_matlab_nl_dataset
from train.semantic_adapter import code_to_nodes
from train.dataset import CodeNLDataset

from train.model import (
    codebert_embed_nodes,
    qwen_embed_text,
    ProjectionMLP,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    num_epochs: int = 3,
    batch_size: int = 1,
    lr: float = 1e-4,
    save_every: int = 100,
    log_every: int = 10,
    max_steps: int = None,  # None = full dataset
):
    print(f"Device: {DEVICE}")
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

    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        for step, batch in enumerate(loader):
            code_nodes = batch["code_nodes"][0]
            nl_text = batch["nl_text"][0]

            z_code = codebert_embed_nodes(code_nodes)   # [768]
            z_target = qwen_embed_text(nl_text)         # [1536]

            z_pred = mlp(z_code)
            loss = torch.nn.functional.mse_loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                print(f"step {global_step} | loss {avg_loss:.4f}")
                running_loss = 0.0

            if save_every and global_step % save_every == 0:
                save_path = f"mlp_checkpoint_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'model_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, save_path)
                print(f"Saved checkpoint to {save_path}")

            if max_steps and global_step >= max_steps:
                print(f"Reached max_steps ({max_steps}), stopping.")
                break

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()

    main(
        num_epochs=args.epochs,
        lr=args.lr,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
    )
