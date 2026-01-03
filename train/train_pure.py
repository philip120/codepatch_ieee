# train/train_pure.py
"""
Pure AST-to-Thought Training

Each AST node → CodeBERT → MLP → One token for Qwen
Qwen sees ALL node tokens at once, then generates pseudocode.
No text hints, no prompts - pure vectors only.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from train.load_dataset import load_matlab_nl_dataset
from train.semantic_adapter import code_to_nodes
from train.dataset import CodeNLDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NodeProjector(nn.Module):
    """Projects CodeBERT node embeddings to Qwen input space."""

    def __init__(self, in_dim: int = 768, out_dim: int = 1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class PureASTModel(nn.Module):
    """
    Pure AST-to-Thought model.

    Each AST node → CodeBERT embedding → Projection → Qwen input token
    Qwen sees all node tokens at once, generates pseudocode.
    """

    def __init__(self, max_nodes: int = 32):
        super().__init__()
        self.max_nodes = max_nodes

        # CodeBERT for node embeddings
        print("Loading CodeBERT...")
        self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")

        # Freeze CodeBERT
        for param in self.codebert.parameters():
            param.requires_grad = False
        self.codebert.eval()

        # Projection layer (TRAINABLE)
        self.projector = NodeProjector(in_dim=768, out_dim=1536)

        # Qwen LLM
        print("Loading Qwen2-1.5B...")
        model_name = "Qwen/Qwen2-1.5B"
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )

        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        # Freeze Qwen
        for param in self.qwen.parameters():
            param.requires_grad = False
        self.qwen.eval()

    def embed_node(self, node_text: str) -> torch.Tensor:
        """Embed a single AST node with CodeBERT."""
        tokens = self.codebert_tokenizer(
            node_text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.codebert(**tokens)

        # Use [CLS] token embedding
        return outputs.last_hidden_state[0, 0, :]  # [768]

    def embed_all_nodes(self, nodes: list[str]) -> torch.Tensor:
        """Embed all AST nodes, return stacked embeddings."""
        embeddings = []
        for node in nodes[:self.max_nodes]:
            emb = self.embed_node(node)
            embeddings.append(emb)

        # Stack: [num_nodes, 768]
        return torch.stack(embeddings, dim=0)

    def forward(self, nodes: list[str], target_text: str):
        """
        Forward pass for training.

        1. Embed each node with CodeBERT
        2. Project to Qwen space
        3. Qwen sees ALL projected nodes, then target text
        4. Compute loss only on target text generation
        """
        # 1. Embed all nodes
        node_embeds = self.embed_all_nodes(nodes)  # [N, 768]

        # 2. Project to Qwen space
        projected = self.projector(node_embeds)  # [N, 1536]
        projected = projected.unsqueeze(0)  # [1, N, 1536]

        # 3. Tokenize target text
        target_tokens = self.qwen_tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(DEVICE)

        # Get target embeddings
        target_embeds = self.qwen.get_input_embeddings()(target_tokens.input_ids)

        # Match dtype
        projected = projected.to(target_embeds.dtype)

        # 4. Concatenate: [all_node_vectors] + [target_text_embeddings]
        #    Qwen sees: [Node₁, Node₂, ..., Nodeₙ, target_word₁, target_word₂, ...]
        input_embeds = torch.cat([projected, target_embeds], dim=1)

        # 5. Attention mask
        num_nodes = projected.shape[1]
        node_mask = torch.ones(1, num_nodes, device=DEVICE)
        attn_mask = torch.cat([node_mask, target_tokens.attention_mask], dim=1)

        # 6. Labels: -100 for node tokens (no loss), target ids for text
        node_labels = torch.full((1, num_nodes), -100, dtype=torch.long, device=DEVICE)
        text_labels = target_tokens.input_ids.clone()
        labels = torch.cat([node_labels, text_labels], dim=1)

        # 7. Forward through Qwen
        outputs = self.qwen(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )

        return outputs.loss

    @torch.no_grad()
    def generate(self, nodes: list[str], max_new_tokens: int = 128) -> str:
        """Generate pseudocode from AST nodes."""
        # Embed and project nodes
        node_embeds = self.embed_all_nodes(nodes)  # [N, 768]
        projected = self.projector(node_embeds)  # [N, 1536]
        projected = projected.unsqueeze(0).to(self.qwen.dtype)  # [1, N, 1536]

        # Generate
        outputs = self.qwen.generate(
            inputs_embeds=projected,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.qwen_tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

        return self.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)


def train(
    num_epochs: int = 10,
    lr: float = 1e-4,
    log_every: int = 10,
    save_every: int = 100,
    eval_every: int = 50,
    max_steps: int = None,
    max_nodes: int = 32,
):
    print(f"Device: {DEVICE}")
    print(f"Max nodes per sample: {max_nodes}")

    # Load dataset
    print("Loading dataset...")
    raw = load_matlab_nl_dataset("train")
    print(f"Loaded {len(raw)} examples")

    dataset = CodeNLDataset(raw, code_to_nodes)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Create model
    model = PureASTModel(max_nodes=max_nodes).to(DEVICE)

    # Only train projector
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr)

    total_steps = num_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # Training
    global_step = 0
    running_loss = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in loader:
            nodes = batch["code_nodes"][0]  # List of node strings
            target = batch["nl_text"][0]    # Target pseudocode

            if len(nodes) == 0:
                continue

            # Forward
            loss = model(nodes, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % log_every == 0:
                avg = running_loss / log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {global_step} | loss {avg:.4f} | lr {lr_now:.2e} | nodes {len(nodes)}")
                running_loss = 0.0

            # Eval sample
            if global_step % eval_every == 0:
                model.projector.eval()
                sample = model.generate(nodes, max_new_tokens=64)
                print(f"  Nodes: {nodes[:3]}...")
                print(f"  Output: {sample[:100]}...")
                print(f"  Target: {target[:100]}...")
                model.projector.train()

            # Save
            if global_step % save_every == 0:
                torch.save({
                    'step': global_step,
                    'projector_state_dict': model.projector.state_dict(),
                    'loss': loss.item(),
                }, f"pure_checkpoint_{global_step}.pt")
                print(f"Saved checkpoint")

            if max_steps and global_step >= max_steps:
                break

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch + 1} | avg_loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'projector_state_dict': model.projector.state_dict(),
                'loss': best_loss,
            }, "pure_best.pt")
            print(f"New best! (loss: {best_loss:.4f})")

        if max_steps and global_step >= max_steps:
            break

    torch.save({
        'projector_state_dict': model.projector.state_dict(),
    }, "pure_final.pt")
    print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_nodes", type=int, default=32)
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        lr=args.lr,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        max_nodes=args.max_nodes,
    )
