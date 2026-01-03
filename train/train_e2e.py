# train/train_e2e.py
"""
End-to-End Training: Code → Thoughts → Frozen LLM → Pseudocode

Train the projection MLP using language modeling loss through a frozen Qwen.
The MLP learns to produce embeddings that help Qwen generate correct pseudocode.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from train.load_dataset import load_matlab_nl_dataset
from train.semantic_adapter import code_to_nodes
from train.dataset import CodeNLDataset
from train.model import codebert_embed_nodes, ProjectionMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CodeToThoughtModel(nn.Module):
    """
    End-to-end model: Code → CodeBERT → MLP → Thought Tokens → Frozen LLM → Text
    """

    def __init__(
        self,
        num_thought_tokens: int = 8,
        freeze_llm: bool = True,
        freeze_codebert: bool = True,
    ):
        super().__init__()

        self.num_thought_tokens = num_thought_tokens

        # Projection MLP (trainable)
        self.mlp = ProjectionMLP(in_dim=768, out_dim=1536, hidden_dim=1024)

        # Thought token expansion (trainable)
        # Learns to create diverse thought tokens from single embedding
        self.thought_expander = nn.Sequential(
            nn.Linear(1536, 1536 * num_thought_tokens),
            nn.GELU(),
        )

        # Load Qwen
        print("Loading Qwen2-1.5B...")
        model_name = "Qwen/Qwen2-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze LLM
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()

        self.freeze_codebert = freeze_codebert

    def get_thought_embeddings(self, code_embedding: torch.Tensor) -> torch.Tensor:
        """
        Convert CodeBERT embedding to thought token embeddings.

        Args:
            code_embedding: [768] from CodeBERT

        Returns:
            thought_embeds: [num_thought_tokens, hidden_dim]
        """
        # Project to LLM space
        projected = self.mlp(code_embedding)  # [1536]

        # Expand to multiple thought tokens
        expanded = self.thought_expander(projected)  # [1536 * num_thoughts]
        thought_embeds = expanded.view(self.num_thought_tokens, -1)  # [num_thoughts, 1536]

        return thought_embeds

    def forward(
        self,
        code_embedding: torch.Tensor,
        target_text: str,
    ):
        """
        Forward pass for training.

        Args:
            code_embedding: [768] CodeBERT embedding of code
            target_text: Ground truth pseudocode string

        Returns:
            loss: Cross-entropy loss on target text generation
            logits: Model output logits
        """
        # Get thought embeddings
        thought_embeds = self.get_thought_embeddings(code_embedding)  # [num_thoughts, 1536]
        thought_embeds = thought_embeds.unsqueeze(0)  # [1, num_thoughts, 1536]

        # Tokenize target text
        target_tokens = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)

        # Get target embeddings
        target_embeds = self.llm.get_input_embeddings()(target_tokens.input_ids)  # [1, seq, 1536]

        # Convert thought embeddings to same dtype
        thought_embeds = thought_embeds.to(target_embeds.dtype)

        # Concatenate: [thoughts] + [target_text]
        input_embeds = torch.cat([thought_embeds, target_embeds], dim=1)

        # Create attention mask
        thought_mask = torch.ones(1, self.num_thought_tokens, device=DEVICE)
        attn_mask = torch.cat([thought_mask, target_tokens.attention_mask], dim=1)

        # Create labels: -100 for thought tokens (don't compute loss), target ids for text
        thought_labels = torch.full((1, self.num_thought_tokens), -100, device=DEVICE)
        text_labels = target_tokens.input_ids.clone()
        labels = torch.cat([thought_labels, text_labels], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )

        return outputs.loss, outputs.logits

    @torch.no_grad()
    def generate(
        self,
        code_embedding: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Generate pseudocode from code embedding."""
        self.llm.eval()

        # Get thought embeddings
        thought_embeds = self.get_thought_embeddings(code_embedding)
        thought_embeds = thought_embeds.unsqueeze(0)  # [1, num_thoughts, 1536]
        thought_embeds = thought_embeds.to(self.llm.dtype)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=thought_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def train(
    num_epochs: int = 10,
    lr: float = 1e-4,
    num_thought_tokens: int = 8,
    log_every: int = 10,
    save_every: int = 100,
    eval_every: int = 50,
    max_steps: int = None,
    grad_accum_steps: int = 4,
):
    print(f"Device: {DEVICE}")
    print(f"Thought tokens: {num_thought_tokens}")
    print(f"Gradient accumulation: {grad_accum_steps}")

    # Load dataset
    print("Loading dataset...")
    raw = load_matlab_nl_dataset("train")
    print(f"Loaded {len(raw)} examples")

    dataset = CodeNLDataset(raw, code_to_nodes)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Create model
    print("Initializing model...")
    model = CodeToThoughtModel(
        num_thought_tokens=num_thought_tokens,
        freeze_llm=True,
        freeze_codebert=True,
    ).to(DEVICE)

    # Only train MLP and thought expander
    trainable_params = list(model.mlp.parameters()) + list(model.thought_expander.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_steps = num_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # Training loop
    global_step = 0
    running_loss = 0.0
    best_loss = float('inf')
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(loader):
            code_nodes = batch["code_nodes"][0]
            nl_text = batch["nl_text"][0]

            # Get CodeBERT embedding
            with torch.no_grad() if model.freeze_codebert else torch.enable_grad():
                code_emb = codebert_embed_nodes(code_nodes)

            # Forward pass
            loss, _ = model(code_emb, nl_text)
            loss = loss / grad_accum_steps

            # Backward
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps
            epoch_loss += loss.item() * grad_accum_steps
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {global_step} | loss {avg_loss:.4f} | lr {lr_now:.2e}")
                running_loss = 0.0

            # Evaluation sample
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    sample_output = model.generate(code_emb, max_new_tokens=64)
                print(f"  Sample output: {sample_output[:100]}...")
                print(f"  Target: {nl_text[:100]}...")
                model.train()

            # Save checkpoint
            if global_step % save_every == 0:
                save_path = f"e2e_checkpoint_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'mlp_state_dict': model.mlp.state_dict(),
                    'expander_state_dict': model.thought_expander.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, save_path)
                print(f"Saved to {save_path}")

            if max_steps and global_step >= max_steps:
                break

        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch + 1} | avg_loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'mlp_state_dict': model.mlp.state_dict(),
                'expander_state_dict': model.thought_expander.state_dict(),
                'loss': best_loss,
            }, "e2e_best.pt")
            print(f"New best model! (loss: {best_loss:.4f})")

        if max_steps and global_step >= max_steps:
            break

    # Final save
    torch.save({
        'step': global_step,
        'mlp_state_dict': model.mlp.state_dict(),
        'expander_state_dict': model.thought_expander.state_dict(),
    }, "e2e_final.pt")
    print(f"\nTraining complete! Saved to e2e_final.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--thoughts", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        lr=args.lr,
        num_thought_tokens=args.thoughts,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
    )
