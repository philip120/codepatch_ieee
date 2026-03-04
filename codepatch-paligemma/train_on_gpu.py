import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
import sys
import matplotlib.pyplot as plt
import types

# We now need the real Gemma model from the transformers library
from transformers import GemmaForCausalLM, GemmaConfig
from peft import get_peft_model, LoraConfig, TaskType

from modeling_codepatch import CodePatchConfig, CodeEncoderConfig, CodePatchForConditionalGeneration
from processing_codepatch import CodePatchProcessor

class PeftCompatibleGemmaConfig(GemmaConfig):
    """A GemmaConfig class that is compatible with PEFT by adding a .get() method."""
    def get(self, key, default=None):
        return getattr(self, key, default)

def get_args():
    parser = argparse.ArgumentParser(description="Train a CodePatch model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="codepatch_checkpoint", help="Directory to save model checkpoints.")
    parser.add_argument("--checkpoint_to_load", type=str, default=None, help="Path to a checkpoint to continue training from.")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients over.")
    return parser.parse_args()

def collate_fn(batch, processor, code_config):
    codes = [item['code'] for item in batch]
    prompts = [item['nl'] for item in batch]

    inputs = processor(code=codes, text=prompts)
    
    labels = inputs["prompt_input_ids"].clone()
    
    # Create ignore labels for the code patch part
    code_patch_ignore_labels = torch.full(
        (len(batch), code_config.num_patches), -100, device=labels.device
    )
    
    # Concatenate ignore labels with the actual prompt labels
    final_labels = torch.cat([code_patch_ignore_labels, labels], dim=1)

    # The model expects labels in the final dict
    inputs['labels'] = final_labels
    return inputs

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Redirect stdout and stderr to a log file ---
    log_file_path = os.path.join(args.output_dir, "training_log.txt")
    sys.stdout = open(log_file_path, 'w')
    sys.stderr = sys.stdout
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gemma_model_name = "google/gemma-2b"

    code_encoder_config = CodeEncoderConfig(freeze_encoder=False)
    processor = CodePatchProcessor(
        "microsoft/codebert-base", gemma_model_name, code_encoder_config, device=device
    )

    # Load the official Gemma config and wrap it in our compatible class
    gemma_config_obj = GemmaConfig.from_pretrained(gemma_model_name)
    text_config = PeftCompatibleGemmaConfig(**gemma_config_obj.to_dict())

    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config),
        text_config=text_config, # Pass the config object directly
        projection_dim=text_config.hidden_size,
        freeze_llm=False, # <-- Unfreeze the LLM for LoRA
    )

    print("Instantiating model and loading REAL Gemma weights...")
    model = CodePatchForConditionalGeneration(full_config).to(device)

    # --- Load Pre-trained Weights BEFORE Applying LoRA ---
    gemma_model = GemmaForCausalLM.from_pretrained(gemma_model_name, torch_dtype=torch.bfloat16)
    model.language_model.load_state_dict(gemma_model.state_dict())
    del gemma_model # Free up memory
    
    # --- Setup PEFT with LoRA ---
    # Monkey-patch the missing method for PEFT compatibility.
    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs
    model.language_model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model.language_model)

    # We freeze all layers first, then apply LoRA which makes only the adapters trainable
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] # Target the query and value projections in the attention layers
    )
    model.language_model = get_peft_model(model.language_model, peft_config)
    
    # Unfreeze the projector and position embeddings so they can be trained alongside the LoRA adapters
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
    for param in model.code_encoder.position_embedding.parameters():
        param.requires_grad = True

    model.language_model.print_trainable_parameters()
    
    # If a checkpoint is provided, load the trained projector and embedding weights
    if args.checkpoint_to_load:
        print(f"Loading projector/embedding weights from: {args.checkpoint_to_load}")
        checkpoint = torch.load(args.checkpoint_to_load, map_location=device)
        model.multi_modal_projector.load_state_dict(checkpoint['projector_state_dict'])
        model.code_encoder.position_embedding.load_state_dict(checkpoint['pos_embedding_state_dict'])
        print("Successfully loaded projector and embedding weights.")
    
    # Note: When using PEFT, we don't need to load the LoRA weights if we are starting fresh
    # But if you were resuming a LoRA training, you would load adapter_model.bin here.

    model.train()

    print("Loading dataset...")
    dataset = load_dataset("philip120/matlab-nl-pseudocode-v2", split="train")
    
    # --- Create a Train/Validation Split ---
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor, code_encoder_config)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor, code_encoder_config)
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = float('inf')

    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        
        for i, batch in enumerate(progress_bar):
            # Move all tensors in the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                logits = outputs["logits"]
                # Shift logits and labels for causal LM loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                # Divide the loss by accumulation steps to average it out
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            scaled_loss = loss / args.accumulation_steps
            scaler.scale(scaled_loss).backward()
            
            # Update weights only every accumulation_steps
            if (i + 1) % args.accumulation_steps == 0:
                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                # Clip the gradients to prevent them from exploding
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() # Log the true, undivided loss
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
            for batch in val_progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    logits = outputs["logits"]
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    val_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                total_val_loss += val_loss.item()
                val_progress_bar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        
        # --- Checkpoint Saving Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best model checkpoint - now we save the full model state
            checkpoint_path = os.path.join(args.output_dir, "best_model_checkpoint.pt")
            model.language_model.save_pretrained(os.path.join(args.output_dir, "lora_adapters"))
            torch.save({
                'projector_state_dict': model.multi_modal_projector.state_dict(),
                'pos_embedding_state_dict': model.code_encoder.position_embedding.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path} and LoRA adapters saved.")


    print("--- Training Complete ---")

    # --- Plotting the training and validation loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), epoch_train_losses, marker='o', linestyle='-', label='Training Loss')
    plt.plot(range(1, args.epochs + 1), epoch_val_losses, marker='o', linestyle='-', label='Validation Loss')
    plt.title("Training & Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, args.epochs + 1))
    plot_path = os.path.join(args.output_dir, "training_loss_plot.png")
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")


if __name__ == "__main__":
    main() 