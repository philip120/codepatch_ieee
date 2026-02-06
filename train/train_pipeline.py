     1|# train/train_pipeline.py
     2|"""
     3|Full Training Pipeline for Semantic ViT
     4|
     5|Usage:
     6|    python -m train.train_pipeline --epochs 10 --batch_size 1 --lr 1e-4
     7|"""
     8|import sys
     9|import os
    10|
    11|sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    12|
    13|import torch
    14|import torch.nn as nn
    15|from torch.utils.data import DataLoader, Dataset
    16|import argparse
    17|from pathlib import Path
    18|
    19|from model.semantic_extractor import SemanticExtractor, MAX_DEPTH, NUM_TYPES
    20|from model.codebert_encoder import CodeBERTEncoder
    21|from model.pixel_embedder import PixelEmbedder
    22|from model.patch_embedder import PatchEmbedder
    23|from model.projector import Projector
    24|from model.qwen_decoder import QwenDecoder
    25|from model.recursive_encoder import RecursiveEncoder
    26|
    27|from train.load_dataset import load_matlab_nl_dataset
    28|from train.matlab_dataset import MatlabPseudocodeDataset
    29|
    30|DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    31|
    32|
    33|# ==============================================================================
    34|# DATASET
    35|# ==============================================================================
    36|
    37|# Dataset class moved to train/matlab_dataset.py
    38|
    39|
    40|# ==============================================================================
    41|# MODEL WRAPPER
    42|# ==============================================================================
    43|
    44|class SemanticViT(nn.Module):
    45|    """
    46|    Full Semantic ViT model wrapping all components.
    47|    Hybrid Architecture:
    48|      - Path 1 (ViT): CodeBERT -> Pixel -> Patch -> Projector
    49|      - Path 2 (RvNN): CodeBERT -> Pixel -> RecursiveEncoder
    50|
    51|    Trainable: PixelEmbedder, Projector, RecursiveEncoder
    52|    Frozen: CodeBERTEncoder, QwenDecoder
    53|    """
    54|
    55|    def __init__(
    56|        self,
    57|        patch_size: int = 4,
    58|        bottleneck_dim: int = 512,
    59|        dropout: float = 0.4,
    60|    ):
    61|        super().__init__()
    62|
    63|        self.patch_size = patch_size
    64|
    65|        # Step 1: Semantic extraction (not nn.Module)
    66|        self.extractor = SemanticExtractor()
    67|
    68|        # Step 2: CodeBERT (frozen, not nn.Module)
    69|        self.encoder = CodeBERTEncoder(device=DEVICE)
    70|
    71|        # Step 3: Pixel embedder (TRAINABLE)
    72|        self.pixel_embedder = PixelEmbedder(
    73|            max_depth=MAX_DEPTH,
    74|            num_types=NUM_TYPES,
    75|        ).to(DEVICE)
    76|
    77|        # Step 4a: Patch embedder (Path 1)
    78|        self.patch_embedder = PatchEmbedder(patch_size=patch_size)
    79|
    80|        # Step 5a: Projector (Path 1)
    81|        self.projector = Projector(
    82|            in_dim=patch_size * 768,
    83|            bottleneck_dim=bottleneck_dim,
    84|            out_dim=1536,
    85|            dropout=dropout,
    86|        ).to(DEVICE)
    87|
    88|        # Step 4b: Recursive Encoder (Path 2)
    89|        # Maps groups of 8 pixels -> 1 pixel recursively
    90|        # Output is 1536 dim (same as Qwen)
    91|        self.recursive_encoder = RecursiveEncoder(
    92|            embed_dim=1536,  # We project pixels to this first
    93|            max_branching=8,
    94|            hidden_dim=3072,
    95|            dropout=dropout
    96|        ).to(DEVICE)
    97|        
    98|        # Project pixels (768) to Recursive dim (1536) before aggregation
    99|        self.pixel_to_recursive = nn.Sequential(
   100|            nn.Linear(768, 1536),
   101|            nn.LayerNorm(1536),
   102|            nn.GELU()
   103|        ).to(DEVICE)
   104|
   105|        # Step 6: Qwen decoder (frozen, not nn.Module)
   106|        self.decoder = QwenDecoder(device=DEVICE)
   107|
   108|    def get_trainable_parameters(self):
   109|        """Return only trainable parameters for optimizer."""
   110|        params = []
   111|        params.extend(self.pixel_embedder.parameters())
   112|        params.extend(self.projector.parameters())
   113|        params.extend(self.recursive_encoder.parameters())
   114|        params.extend(self.pixel_to_recursive.parameters())
   115|        return params
   116|
   117|    def num_trainable_parameters(self):
   118|        return sum(p.numel() for p in self.get_trainable_parameters())
   119|
   120|    def build_recursive_tree(self, ops, pixel_embeddings):
   121|        """
   122|        Manually build the recursive tree from linear pixels.
   123|        Naive implementation: Just chunks pixels linearly for now.
   124|        Ideally: Use the 'ops' tree structure.
   125|        
   126|        Current Logic (Simplified):
   127|        1. Project all pixels: [N, 768] -> [N, 1536]
   128|        2. Aggregate 8-blocks recursively until 1 vector remains.
   129|        """
   130|        # 1. Project to 1536
   131|        # [N, 768] -> [N, 1536]
   132|        current_vectors = self.pixel_to_recursive(pixel_embeddings)
   133|        current_vectors = current_vectors.unsqueeze(0) # Batch dim: [1, N, 1536]
   134|        
   135|        # 2. Recursive Aggregation Loop
   136|        # Loop until we have 1 vector (or small number)
   137|        while current_vectors.size(1) > 1:
   138|            N = current_vectors.size(1)
   139|            num_blocks = (N + 7) // 8 # Ceil division
   140|            
   141|            next_level_vectors = []
   142|            for i in range(num_blocks):
   143|                # Get chunk of 8
   144|                chunk = current_vectors[:, i*8 : (i+1)*8, :] # [1, k, 1536]
   145|                
   146|                # Aggregate
   147|                # [1, 1536]
   148|                block_vec = self.recursive_encoder(chunk)
   149|                next_level_vectors.append(block_vec)
   150|            
   151|            # Stack for next iteration
   152|            # [1, num_blocks, 1536]
   153|            current_vectors = torch.stack(next_level_vectors, dim=1).squeeze(2)
   154|            
   155|        # Final result: [1, 1536] (remove batch if needed by caller, but we need it for concat)
   156|        return current_vectors.squeeze(0) # [1, 1536]
   157|
   158|    def forward(self, code: str, target: str = None, features: dict = None):
   159|        """
   160|        Forward pass.
   161|        """
   162|        # Step 1: Extract semantic features
   163|        if features is None:
   164|            features = self.extractor(code)
   165|
   166|        if not features['texts']:
   167|            if target:
   168|                return torch.tensor(0.0, device=DEVICE, requires_grad=True)
   169|            return None
   170|
   171|        # Step 2: CodeBERT embeddings
   172|        cls_embeddings = self.encoder(features['texts'])
   173|
   174|        # Step 3: Pixel embeddings
   175|        depth_ids = torch.tensor(features['depths'], device=DEVICE)
   176|        type_ids = torch.tensor(features['type_ids'], device=DEVICE)
   177|        pixel_embeddings = self.pixel_embedder(cls_embeddings, depth_ids, type_ids)
   178|
   179|        # --- PATH 1: Standard ViT ---
   180|        # [N, 768] -> [Patches, 3072] -> [Patches, 1536]
   181|        patch_embeddings = self.patch_embedder(pixel_embeddings)
   182|        projected_patches = self.projector(patch_embeddings)
   183|        
   184|        # --- PATH 2: Recursive Structure ---
   185|        # [N, 768] -> [1, 1536] Global Context
   186|        global_vector = self.build_recursive_tree(features.get('ops'), pixel_embeddings)
   187|        
   188|        # Combine: [Global (1), Patches (M)]
   189|        # global_vector is [1, 1536]
   190|        # projected_patches is [M, 1536]
   191|        combined_embeddings = torch.cat([global_vector, projected_patches], dim=0)
   192|
   193|        # Step 6: Decoder
   194|        if target:
   195|            # Training: compute loss
   196|            loss = self.decoder.forward_train(combined_embeddings, target)
   197|            return loss
   198|        else:
   199|            # Inference
   200|            return combined_embeddings
   201|
   202|    @torch.no_grad()
   203|    def generate(self, code: str, max_new_tokens: int = 128) -> str:
   204|        """Generate pseudocode from MATLAB code."""
   205|        projected = self.forward(code, target=None)
   206|        if projected is None:
   207|            return ""
   208|        return self.decoder.generate(projected, max_new_tokens=max_new_tokens)
   209|
   210|
   211|# ==============================================================================
   212|# TRAINING LOOP
   213|# ==============================================================================
   214|
   215|def train(
   216|    split: str = "train",
   217|    epochs: int = 10,
   218|    lr: float = 1e-4,
   219|    weight_decay: float = 0.05,
   220|    patch_size: int = 4,
   221|    bottleneck_dim: int = 512,
   222|    dropout: float = 0.4,
   223|    log_every: int = 10,
   224|    eval_every: int = 50,
   225|    save_every: int = 100,
   226|    save_dir: str = "checkpoints",
   227|    gradient_accumulation: int = 4,
   228|):
   229|    """
   230|    Main training function.
   231|    """
   232|    print("=" * 60)
   233|    print("SEMANTIC VIT TRAINING (HYBRID MODE)")
   234|    print("=" * 60)
   235|    print(f"Device: {DEVICE}")
   236|    print(f"Epochs: {epochs}")
   237|    print(f"Learning rate: {lr}")
   238|    print(f"Weight decay: {weight_decay}")
   239|    print(f"Patch size: {patch_size}")
   240|    print(f"Bottleneck: {bottleneck_dim}")
   241|    print(f"Dropout: {dropout}")
   242|    print(f"Gradient accumulation: {gradient_accumulation}")
   243|
   244|    # Create save directory
   245|    save_path = Path(save_dir)
   246|    save_path.mkdir(exist_ok=True)
   247|
   248|    # Load dataset from Hugging Face
   249|    print("\n" + "=" * 60)
   250|    print("Loading dataset from Hugging Face...")
   251|    dataset = MatlabPseudocodeDataset(split=split)
   252|    # Use collate_fn to return single sample dict directly (batch_size=1)
   253|    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
   254|
   255|    if len(dataset) == 0:
   256|        print("ERROR: No samples found!")
   257|        return
   258|
   259|    # Create model
   260|    print("\n" + "=" * 60)
   261|    print("Creating model...")
   262|    model = SemanticViT(
   263|        patch_size=patch_size,
   264|        bottleneck_dim=bottleneck_dim,
   265|        dropout=dropout,
   266|    )
   267|
   268|    print(f"\nTrainable parameters: {model.num_trainable_parameters():,}")
   269|
   270|    # Optimizer
   271|    optimizer = torch.optim.AdamW(
   272|        model.get_trainable_parameters(),
   273|        lr=lr,
   274|        weight_decay=weight_decay,
   275|    )
   276|
   277|    # Learning rate scheduler
   278|    total_steps = epochs * len(loader) // gradient_accumulation
   279|    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
   280|        optimizer,
   281|        T_max=total_steps,
   282|        eta_min=1e-6,
   283|    )
   284|
   285|    # Training loop
   286|    print("\n" + "=" * 60)
   287|    print("Starting training...")
   288|    print("=" * 60)
   289|
   290|    global_step = 0
   291|    accumulation_step = 0
   292|    running_loss = 0.0
   293|    best_loss = float('inf')
   294|
   295|    model.pixel_embedder.train()
   296|    model.projector.train()
   297|    model.recursive_encoder.train()
   298|    model.pixel_to_recursive.train()
   299|
   300|    for epoch in range(epochs):
   301|        print(f"\n{'='*60}")
   302|        print(f"EPOCH {epoch + 1}/{epochs}")
   303|        print(f"{'='*60}")
   304|
   305|        epoch_loss = 0.0
   306|        epoch_samples = 0
   307|
   308|        for batch_idx, batch in enumerate(loader):
   309|            code = batch['code']
   310|            target = batch['target']
   311|            features = batch.get('features')
   312|
   313|            # Forward pass
   314|            loss = model(code, target=target, features=features)
   315|
   316|            if loss is None or loss.item() == 0:
   317|                continue
   318|
   319|            # Scale loss for gradient accumulation
   320|            loss = loss / gradient_accumulation
   321|            loss.backward()
   322|
   323|            accumulation_step += 1
   324|            running_loss += loss.item() * gradient_accumulation
   325|            epoch_loss += loss.item() * gradient_accumulation
   326|            epoch_samples += 1
   327|
   328|            # Update weights after accumulation
   329|            if accumulation_step % gradient_accumulation == 0:
   330|                torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
   331|                optimizer.step()
   332|                scheduler.step()
   333|                optimizer.zero_grad()
   334|                global_step += 1
   335|
   336|                # Logging
   337|                if global_step % log_every == 0:
   338|                    avg_loss = running_loss / (log_every * gradient_accumulation)
   339|                    lr_now = scheduler.get_last_lr()[0]
   340|                    print(f"  step {global_step:4d} | loss {avg_loss:.4f} | lr {lr_now:.2e}")
   341|                    running_loss = 0.0
   342|
   343|                # Evaluation
   344|                if global_step % eval_every == 0:
   345|                    model.pixel_embedder.eval()
   346|                    model.projector.eval()
   347|                    model.recursive_encoder.eval()
   348|                    model.pixel_to_recursive.eval()
   349|
   350|                    print(f"\n  --- Eval at step {global_step} ---")
   351|                    print(f"  Code: {code[:80]}...")
   352|                    print(f"  Target: {target[:80]}...")
   353|
   354|                    generated = model.generate(code, max_new_tokens=64)
   355|                    print(f"  Generated: {generated[:80]}...")
   356|                    print()
   357|
   358|                    model.pixel_embedder.train()
   359|                    model.projector.train()
   360|                    model.recursive_encoder.train()
   361|                    model.pixel_to_recursive.train()
   362|
   363|                # Save checkpoint
   364|                if global_step % save_every == 0:
   365|                    checkpoint = {
   366|                        'step': global_step,
   367|                        'epoch': epoch,
   368|                        'pixel_embedder': model.pixel_embedder.state_dict(),
   369|                        'projector': model.projector.state_dict(),
   370|                        'recursive_encoder': model.recursive_encoder.state_dict(),
   371|                        'pixel_to_recursive': model.pixel_to_recursive.state_dict(),
   372|                        'optimizer': optimizer.state_dict(),
   373|                        'scheduler': scheduler.state_dict(),
   374|                        'loss': epoch_loss / max(epoch_samples, 1),
   375|                    }
   376|                    torch.save(checkpoint, save_path / f"checkpoint_{global_step}.pt")
   377|                    print(f"  Saved checkpoint_{global_step}.pt")
   378|
   379|        # End of epoch
   380|        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
   381|        print(f"\nEpoch {epoch + 1} complete | avg_loss: {avg_epoch_loss:.4f}")
   382|
   383|        # Save best model
   384|        if avg_epoch_loss < best_loss:
   385|            best_loss = avg_epoch_loss
   386|            checkpoint = {
   387|                'step': global_step,
   388|                'epoch': epoch,
   389|                'pixel_embedder': model.pixel_embedder.state_dict(),
   390|                'projector': model.projector.state_dict(),
   391|                'recursive_encoder': model.recursive_encoder.state_dict(),
   392|                'pixel_to_recursive': model.pixel_to_recursive.state_dict(),
   393|                'loss': best_loss,
   394|            }
   395|            torch.save(checkpoint, save_path / "best_model.pt")
   396|            print(f"  New best model! Loss: {best_loss:.4f}")
   397|
   398|    print("\n" + "=" * 60)
   399|    print("TRAINING COMPLETE")
   400|    print(f"Best loss: {best_loss:.4f}")
   401|    print(f"Checkpoints saved to: {save_path}")
   402|    print("=" * 60)
   403|
   404|
   405|# ==============================================================================
   406|# MAIN
   407|# ==============================================================================
   408|
   409|if __name__ == "__main__":
   410|    parser = argparse.ArgumentParser(description="Train Semantic ViT")
   411|
   412|    parser.add_argument("--split", type=str, default="train",
   413|                        help="Dataset split (train/test)")
   414|    parser.add_argument("--epochs", type=int, default=10)
   415|    parser.add_argument("--lr", type=float, default=1e-4)
   416|    parser.add_argument("--weight_decay", type=float, default=0.05)
   417|    parser.add_argument("--patch_size", type=int, default=4)
   418|    parser.add_argument("--bottleneck", type=int, default=512)
   419|    parser.add_argument("--dropout", type=float, default=0.4)
   420|    parser.add_argument("--log_every", type=int, default=10)
   421|    parser.add_argument("--eval_every", type=int, default=50)
   422|    parser.add_argument("--save_every", type=int, default=100)
   423|    parser.add_argument("--save_dir", type=str, default="checkpoints")
   424|    parser.add_argument("--grad_accum", type=int, default=4)
   425|
   426|    args = parser.parse_args()
   427|
   428|    train(
   429|        split=args.split,
   430|        epochs=args.epochs,
   431|        lr=args.lr,
   432|        weight_decay=args.weight_decay,
   433|        patch_size=args.patch_size,
   434|        bottleneck_dim=args.bottleneck,
   435|        dropout=args.dropout,
   436|        log_every=args.log_every,
   437|        eval_every=args.eval_every,
   438|        save_every=args.save_every,
   439|        save_dir=args.save_dir,
   440|        gradient_accumulation=args.grad_accum,
   441|    )
   442|