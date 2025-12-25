# train/train_loop.py
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

def main():
    raw = load_matlab_nl_dataset("train")
    dataset = CodeNLDataset(raw, code_to_nodes)

    loader = DataLoader(
        dataset,
        batch_size=1,   # start with 1
        shuffle=True
    )

    mlp = ProjectionMLP(
        in_dim=768,
        out_dim=1536
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-4)

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

        print(f"step {step} | loss {loss.item():.4f}")

        if step >= 10:
            break

if __name__ == "__main__":
    main()
