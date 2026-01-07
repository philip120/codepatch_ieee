# train/train_patch.py
"""
Patch-based Training Strategy

Divides AST nodes into fixed-size "patches".
A specific MLP aggregates each patch of nodes into a single vector.
This reduces sequence length while preserving local structure.

Flow:
1. AST Nodes -> CodeBERT Embeddings [N, 768]
2. Group into Patches [N/P, P, 768]
3. Flatten Patches [N/P, P*768]
4. MLP Projector -> [N/P, 1536]
5. Qwen LLM -> Pseudocode
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


class PatchProjector(nn.Module):
    """
    Aggregates a patch of P nodes into a single vector.
    
    This MLP acts as the "translator" or "compressor".
    It takes a sequence of P CodeBERT embeddings (representing a local window of code)
    and fuses them into a single embedding vector that resides in Qwen's latent space.
    
    Input:  Flattened patch vector of shape [batch_size, patch_size * 768]
    Output: Projected vector of shape [batch_size, 1536] (Qwen's embedding dim)
    """

    def __init__(self, patch_size: int = 4, in_dim: int = 768, out_dim: int = 1536):
        super().__init__()
        self.patch_size = patch_size
        # The input dimension is the concatenation of all node embeddings in the patch
        # e.g., 4 * 768 = 3072 dimensions
        self.flat_in_dim = patch_size * in_dim
        
        # A 2-layer MLP with GELU activation and LayerNorm
        # This structure allows for non-linear interactions between nodes in the patch.
        # It learns to say "If Node1 is 'if' AND Node2 is 'x>0', output Vector Y".
        self.net = nn.Sequential(
            nn.Linear(self.flat_in_dim, self.flat_in_dim // 2),  # Compression Step 1
            nn.GELU(),                                           # Non-linearity
            nn.LayerNorm(self.flat_in_dim // 2),                 # Stabilization
            nn.Linear(self.flat_in_dim // 2, out_dim),           # Projection to Qwen space
            nn.LayerNorm(out_dim),                               # Final Norm
        )

    def forward(self, x):
        """
        Forward pass of the projector.
        x: [batch_size, patch_size * 768]
        returns: [batch_size, 1536]
        """
        return self.net(x)


class PatchModel(nn.Module):
    """
    End-to-End Patch-Based Model.
    
    Architecture:
    1. Code Nodes -> CodeBERT -> [N, 768]
    2. Grouping -> Patches of size P -> [N/P, P*768]
    3. PatchProjector -> [N/P, 1536]
    4. Qwen LLM -> Generates Pseudocode from these patch embeddings
    """
    def __init__(self, patch_size: int = 4, max_nodes: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.max_nodes = max_nodes
        
        # Ensure max_nodes is divisible by patch_size for simplicity.
        # This guarantees we always have full patches (after padding).
        if self.max_nodes % self.patch_size != 0:
            self.max_nodes = ((self.max_nodes // self.patch_size) + 1) * self.patch_size

        # --- 1. Encoder (Frozen) ---
        # We use CodeBERT to get the initial vector representation of each AST node.
        # We freeze it to save memory and because it's already a good code encoder.
        print("Loading CodeBERT...")
        self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        
        for param in self.codebert.parameters():
            param.requires_grad = False
        self.codebert.eval()

        # --- 2. Projector (Trainable) ---
        # This is the ONLY part of the model that learns.
        # It learns to map CodeBERT patches -> Qwen tokens.
        self.projector = PatchProjector(
            patch_size=patch_size, 
            in_dim=768, 
            out_dim=1536
        )

        # --- 3. Decoder / LLM (Frozen) ---
        # We use Qwen2-1.5B as the "brain" to generate text.
        # We freeze it so we don't destroy its language capabilities.
        print("Loading Qwen2-1.5B...")
        model_name = "Qwen/Qwen2-1.5B"
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        for param in self.qwen.parameters():
            param.requires_grad = False
        self.qwen.eval()

    def embed_nodes(self, nodes: list[str]) -> torch.Tensor:
        """
        Embeds nodes and groups them into patches.
        
        Process:
        1. Pad the list of nodes so its length is a multiple of patch_size.
        2. Run CodeBERT on all nodes to get [N_padded, 768].
        3. Reshape/View the tensor to group vectors into patches.
        
        Returns: [num_patches, patch_size * 768]
                 This is a list of "flattened patches" ready for the MLP.
        """
        # 1. Pad nodes to multiple of patch_size
        # Clip to max length first to avoid OOM
        nodes = nodes[:self.max_nodes]
        current_len = len(nodes)
        
        # Calculate padding needed
        # e.g. if len=10 and patch=4, we need 2 more to get to 12.
        remainder = current_len % self.patch_size
        pad_len = 0
        if remainder != 0:
            pad_len = self.patch_size - remainder
        elif current_len == 0:
            pad_len = self.patch_size
            
        # Add empty strings as padding (CodeBERT will embed these as something neutral/CLS)
        padded_nodes = nodes + [""] * pad_len
        
        # 2. Embed all nodes individually with CodeBERT
        # We process in batches via the tokenizer for speed
        inputs = self.codebert_tokenizer(
            padded_nodes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.codebert(**inputs)
            # Use CLS token [N, 768] as the node representation
            node_embeds = outputs.last_hidden_state[:, 0, :]
            
        # 3. Reshape into patches [num_patches, patch_size * 768]
        # view() effectively "concatenates" the vectors of the patch into one long row
        num_patches = len(padded_nodes) // self.patch_size
        patches = node_embeds.view(num_patches, self.patch_size * 768)
        
        return patches

    def forward(self, nodes: list[str], target_text: str):
        """
        Forward pass for Training.
        
        1. Convert Nodes -> Patch Embeddings (via Projector)
        2. Convert Target Text -> Target Embeddings (via Qwen)
        3. Concatenate: [Patch Embeddings] + [Target Embeddings]
        4. Feed to Qwen to calculate "Next Token Prediction" loss.
        """
        # 1. Get patch embeddings [num_patches, input_dim]
        # This runs CodeBERT (frozen)
        patch_inputs = self.embed_nodes(nodes)
        
        # 2. Project patches [num_patches, 1536]
        # This runs the MLP (Trainable) - gradients start here
        projected = self.projector(patch_inputs)
        projected = projected.unsqueeze(0) # Batch size 1: [1, num_patches, 1536]
        
        # 3. Tokenize target text (The ground truth pseudocode)
        target_tokens = self.qwen_tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(DEVICE)
        
        # Get standard embeddings for the text part
        target_embeds = self.qwen.get_input_embeddings()(target_tokens.input_ids)
        
        # Match dtype (often float16 for Qwen)
        projected = projected.to(target_embeds.dtype)
        
        # 4. Concatenate: [Patches] + [Target]
        # This forms the full context window for the LLM
        input_embeds = torch.cat([projected, target_embeds], dim=1)
        
        # 5. Create Attention Mask
        # 1s for patches, and the original mask for the text part
        num_patches = projected.shape[1]
        patch_mask = torch.ones(1, num_patches, device=DEVICE)
        attn_mask = torch.cat([patch_mask, target_tokens.attention_mask], dim=1)
        
        # 6. Create Labels for Loss Calculation
        # -100 means "ignore this position in loss calculation"
        # We ignore the patch tokens (we don't predict them)
        # We predict the text tokens
        patch_labels = torch.full((1, num_patches), -100, dtype=torch.long, device=DEVICE)
        text_labels = target_tokens.input_ids.clone()
        labels = torch.cat([patch_labels, text_labels], dim=1)
        
        # 7. Forward through Qwen
        # This calculates the standard Causal LM Loss
        outputs = self.qwen(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels
        )
        
        return outputs.loss

    @torch.no_grad()
    def generate(self, nodes: list[str], max_new_tokens: int = 128) -> str:
        """
        Inference / Generation.
        
        1. Embed Nodes -> Patches -> Projected Vectors
        2. Feed Projected Vectors as the "Prompt" to Qwen
        3. Auto-regressively generate the text
        """
        patch_inputs = self.embed_nodes(nodes)
        projected = self.projector(patch_inputs)
        projected = projected.unsqueeze(0).to(self.qwen.dtype)
        
        outputs = self.qwen.generate(
            inputs_embeds=projected,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.qwen_tokenizer.eos_token_id,
        )
        
        return self.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)


def train(
    num_epochs: int = 10,
    lr: float = 1e-4,
    patch_size: int = 4,
    max_nodes: int = 64,
    log_every: int = 10,
    save_every: int = 100,
    eval_every: int = 50,
    max_steps: int = None,
):
    print(f"Device: {DEVICE}")
    print(f"Patch Size: {patch_size}")
    print(f"Max Nodes: {max_nodes}")

    # Dataset
    print("Loading dataset...")
    raw = load_matlab_nl_dataset("train")
    dataset = CodeNLDataset(raw, code_to_nodes)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Model
    model = PatchModel(patch_size=patch_size, max_nodes=max_nodes).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr)
    
    total_steps = num_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    
    global_step = 0
    running_loss = 0.0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch in loader:
            nodes = batch["code_nodes"][0]
            target = batch["nl_text"][0]
            
            if len(nodes) == 0:
                continue
                
            loss = model(nodes, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            if global_step % log_every == 0:
                avg = running_loss / log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {global_step} | loss {avg:.4f} | lr {lr_now:.2e}")
                running_loss = 0.0
                
            if global_step % eval_every == 0:
                model.projector.eval()
                print(f"  Nodes: {len(nodes)}")
                sample = model.generate(nodes, max_new_tokens=64)
                print(f"  Output: {sample[:100]}...")
                print(f"  Target: {target[:100]}...")
                model.projector.train()
                
            if global_step % save_every == 0:
                torch.save({
                    'step': global_step,
                    'projector_state_dict': model.projector.state_dict(),
                }, f"patch_checkpoint_{global_step}.pt")
                
            if max_steps and global_step >= max_steps:
                break
                
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch + 1} | avg_loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'projector_state_dict': model.projector.state_dict(),
                'loss': best_loss,
            }, "patch_best.pt")
            print(f"New best! {best_loss:.4f}")

        if max_steps and global_step >= max_steps:
            break
            
    print("Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--max_nodes", type=int, default=64)
    args = parser.parse_args()
    
    train(
        num_epochs=args.epochs,
        patch_size=args.patch_size,
        max_nodes=args.max_nodes
    )

