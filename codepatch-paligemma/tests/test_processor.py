import torch

from modeling_codepatch import CodeEncoderConfig
from processing_codepatch import CodePatchProcessor


def main():
    print("--- Testing CodePatchProcessor ---")

    # --- 1. Setup ---
    # Instantiate the config needed by the processor
    code_config = CodeEncoderConfig()

    # In a real scenario, the text_tokenizer would be specific to the LLM (e.g., Gemma's).
    # For this test, we can reuse the code tokenizer as a placeholder.
    code_tokenizer_name = "microsoft/codebert-base"
    text_tokenizer_name = "microsoft/codebert-base" 

    print("Instantiating processor...")
    processor = CodePatchProcessor(code_tokenizer_name, text_tokenizer_name, code_config)
    print("Processor instantiated successfully.")

    # --- 2. Load Data ---
    matlab_file_path = "test.m"
    print(f"\nLoading sample code from: {matlab_file_path}")
    with open(matlab_file_path, "r") as f:
        code_string = f.read()
    
    prompt_string = "What does this MATLAB code do?"
    print(f"Using sample prompt: '{prompt_string}'")

    # --- 3. Process Data ---
    print("\nCalling processor to transform data...")
    model_inputs = processor(code=code_string, text=prompt_string)
    print("Processing complete.")

    # --- 4. Verify Output ---
    print("\n--- Verifying Processor Output ---")
    
    expected_keys = [
        "code_input_ids",
        "code_attention_mask",
        "prompt_input_ids",
        "prompt_attention_mask",
    ]
    
    print("Checking for expected keys...")
    all_keys_present = True
    for key in expected_keys:
        if key not in model_inputs:
            print(f"FAILED: Missing key '{key}' in processor output.")
            all_keys_present = False
    
    if not all_keys_present:
        return

    print("All expected keys are present. PASSED.")
    
    print("\nChecking tensor shapes...")
    code_ids_shape = model_inputs["code_input_ids"].shape
    code_mask_shape = model_inputs["code_attention_mask"].shape
    prompt_ids_shape = model_inputs["prompt_input_ids"].shape
    prompt_mask_shape = model_inputs["prompt_attention_mask"].shape

    print(f"Shape of 'code_input_ids': {code_ids_shape}")
    print(f"Shape of 'code_attention_mask': {code_mask_shape}")
    print(f"Shape of 'prompt_input_ids': {prompt_ids_shape}")
    print(f"Shape of 'prompt_attention_mask': {prompt_mask_shape}")

    # Assertions to formalize the check
    assert code_ids_shape == torch.Size([1, code_config.num_patches, code_config.patch_length])
    assert code_mask_shape == torch.Size([1, code_config.num_patches, code_config.patch_length])
    assert prompt_ids_shape[0] == 1
    assert prompt_ids_shape == prompt_mask_shape

    print("\nAll tensor shapes are correct. PASSED.")
    print("\n--- Processor test successfully completed! ---")


if __name__ == "__main__":
    main() 