import torch

from modeling_codepatch import (
    CodePatchConfig,
    CodeEncoderConfig,
    CodePatchForConditionalGeneration,
)
from processing_codepatch import CodePatchProcessor


def main():
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Configuration ---
    print("Creating model configurations...")
    code_encoder_config = CodeEncoderConfig()

    text_config_dict = {
        "vocab_size": 256000,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 1024,
    }

    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config),
        text_config=text_config_dict,
        projection_dim=text_config_dict["hidden_size"],
    )
    print("Configurations created successfully.")

    # --- 3. Model Instantiation ---
    print("Instantiating the full CodePatchForConditionalGeneration model...")
    model = CodePatchForConditionalGeneration(full_config).to(device).eval()
    print("Model instantiated successfully.")

    # --- 4. Prepare Inputs ---
    print("Preparing code patches and prompt tensors using CodePatchProcessor...")
    processor = CodePatchProcessor(
        code_tokenizer_name=code_encoder_config.model_name_or_path,
        text_tokenizer_name="bert-base-uncased",  # Placeholder, should be Gemma's
        config=code_encoder_config,
        device=device,
    )

    matlab_file_path = "../test.m"
    with open(matlab_file_path, "r") as f:
        matlab_code = f.read()

    prompt_text = "What is the title of this plot?"
    inputs = processor(code=matlab_code, text=prompt_text)

    print(f"Shape of code_input_ids: {inputs['code_input_ids'].shape}")
    print(f"Shape of prompt_input_ids: {inputs['prompt_input_ids'].shape}")

    # --- 5. Run Forward Pass ---
    print("\nRunning the forward pass through the integrated model...")
    with torch.no_grad():
        outputs = model(**inputs)

    # --- 6. Check Output ---
    print("Forward pass complete.")
    logits = outputs["logits"]
    print(f"Shape of the final output logits: {logits.shape}")
    
    expected_seq_len = inputs["code_input_ids"].shape[1] + inputs["prompt_input_ids"].shape[1]
    print(f"Expected sequence length: {expected_seq_len}")
    print(f"Expected vocabulary size: {text_config_dict['vocab_size']}")

    assert logits.shape[0] == 1
    assert logits.shape[1] == expected_seq_len
    assert logits.shape[2] == text_config_dict['vocab_size']

    print("\nTest passed! The model architecture is sound and data flows correctly.")


if __name__ == "__main__":
    main() 