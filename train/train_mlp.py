# train/train_mlp.py
import torch
import torch.nn.functional as F

def train_epoch(dataset, mlp, optimizer, embed_code, embed_nl):
    total_loss = 0.0

    for ex in dataset:
        z_code = embed_code(ex["code_nodes"])
        z_pred = mlp(z_code)

        with torch.no_grad():
            z_target = embed_nl(ex["nl_text"])

        loss = 1 - F.cosine_similarity(
            z_pred.unsqueeze(0),
            z_target.unsqueeze(0)
        ).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataset)
