import torch
from transformers import AutoTokenizer
from typing import List

from modeling_codepatch import CodePatchModel, CodeEncoderConfig
from ast_parser.matlab_parser import MatlabParser
from ast_parser.matlab_lexer import MatlabLexer
from ast_parser.ast_utils import get_semantic_patches


def process_code_for_encoder_ast(
    semantic_patches: List[str], 
    tokenizer, 
    config: CodeEncoderConfig, 
    device: str
):
    """
    Processes a list of semantic patches into tokenized and padded tensors.
    """
    # Tokenize each patch individually
    tokenized_patches = tokenizer(
        semantic_patches,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.patch_length,
    )["input_ids"]

    num_real_patches = len(tokenized_patches)
    
    # Pad patches if necessary to reach config.num_patches
    num_padding_patches = config.num_patches - num_real_patches
    if num_padding_patches < 0:
        tokenized_patches = tokenized_patches[:config.num_patches]
        num_padding_patches = 0
        num_real_patches = config.num_patches

    padding = torch.full((num_padding_patches, config.patch_length), tokenizer.pad_token_id, dtype=torch.long)
    final_patches = torch.cat((tokenized_patches, padding), dim=0)

    # Create attention mask
    attention_mask = (final_patches != tokenizer.pad_token_id).long()
    
    # Add a batch dimension and send to device
    final_patches = final_patches.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    return final_patches, attention_mask, num_real_patches


def main():
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the tokenizer for our code encoder
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # Instantiate our model's configuration
    config = CodeEncoderConfig()

    # Instantiate the CodePatchModel
    model = CodePatchModel(config).to(device).eval()
    print("CodePatchModel loaded successfully.")
    
    # Instantiate the MATLAB parser and lexer
    parser = MatlabParser()
    lexer = MatlabLexer()

    # --- 2. Load and Process Data ---
    matlab_file_path = "test.m"
    print(f"Loading MATLAB code from: {matlab_file_path}")
    with open(matlab_file_path, "r") as f:
        matlab_code = f.read()

    # Clean the code by replacing newlines with spaces for the parser
    cleaned_code = matlab_code.replace('\\n', ' ').replace('\n', ' ')
    
    print("Generating semantic patches using AST...")
    semantic_patches = get_semantic_patches(cleaned_code, parser, lexer)

    print(f"Generated {len(semantic_patches)} semantic patches:")
    for i, patch in enumerate(semantic_patches):
        print(f"  Patch {i+1}: {patch}")

    print("\nProcessing patches for the model...")
    input_patches, attention_mask, num_real_patches = process_code_for_encoder_ast(
        semantic_patches, tokenizer, config, device
    )

    print(f"Shape of input patches tensor: {input_patches.shape}")
    print(f"Shape of attention mask tensor: {attention_mask.shape}")

    # --- 3. Run Inference ---
    print("\nFeeding patches to the model...")
    with torch.no_grad():
        output_vectors = model(input_ids=input_patches, attention_mask=attention_mask)

    # --- 4. Display Results ---
    print("Inference complete.")
    print(f"Shape of the output vectors: {output_vectors.shape}")
    print(f"Number of actual code patches created from the file: {num_real_patches}")

    print("\n--- Summary vectors for each non-padding patch ---")
    for i in range(num_real_patches):
        patch_vector = output_vectors[0, i]
        # Printing the first 8 dimensions for brevity
        print(f"Patch {i+1:02d} summary vector (first 8 dims): {patch_vector[:8].tolist()}")

    if num_real_patches < config.num_patches:
        print(f"\nPatches {num_real_patches + 1} through {config.num_patches} are padding and were not shown.")


if __name__ == "__main__":
    main() 