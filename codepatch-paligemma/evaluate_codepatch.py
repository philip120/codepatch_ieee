import torch
import argparse
import warnings
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from langdetect import detect, LangDetectException
import types
warnings.filterwarnings("ignore", category=FutureWarning)

from modeling_codepatch import CodePatchConfig, CodeEncoderConfig, CodePatchForConditionalGeneration
from processing_codepatch import CodePatchProcessor
from transformers import GemmaForCausalLM, GemmaConfig
from peft import PeftModel

from ast_parser.ast_utils import get_semantic_patches

class PeftCompatibleGemmaConfig(GemmaConfig):
    """A GemmaConfig class that is compatible with PEFT by adding a .get() method."""
    def get(self, key, default=None):
        return getattr(self, key, default)

def get_args():
    parser = argparse.ArgumentParser(description="Run evaluation with a trained CodePatch model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the projector/embedding checkpoint .pt file.")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the trained LoRA adapter directory.")
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="Path to the evaluation dataset JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repeating tokens.")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints for inputs and per-token generation.")
    return parser.parse_args()

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gemma_model_name = "google/gemma-2b"

    # --- 1. Re-create the exact model and processor configuration used for training ---
    code_encoder_config = CodeEncoderConfig(freeze_encoder=True, num_patches=64)
    processor = CodePatchProcessor(
        "microsoft/codebert-base", gemma_model_name, code_encoder_config, device=device
    )
    gemma_config_obj = GemmaConfig.from_pretrained(gemma_model_name)
    text_config = PeftCompatibleGemmaConfig(**gemma_config_obj.to_dict())
    
    # --- We'll need these config values for the KV cache calculation ---
    num_layers = text_config.num_hidden_layers
    hidden_size = text_config.hidden_size
    bytes_per_param = 2 # for bfloat16

    full_config = CodePatchConfig(
        code_encoder_config=vars(code_encoder_config), text_config=text_config, # Pass object directly
        projection_dim=text_config.hidden_size, freeze_llm=True,
    )

    # --- 2. Instantiate the model and load the trained weights ---
    print("Instantiating model and loading REAL Gemma weights...")
    model = CodePatchForConditionalGeneration(full_config).to(device)
    
    # Monkey-patch the missing method for PEFT compatibility
    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs
    model.language_model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, model.language_model)

    gemma_model = GemmaForCausalLM.from_pretrained(gemma_model_name, torch_dtype=torch.bfloat16)
    model.language_model.load_state_dict(gemma_model.state_dict())

    # --- Load LoRA Adapters ---
    print(f"Loading LoRA adapters from: {args.lora_adapter_path}")
    model.language_model = PeftModel.from_pretrained(model.language_model, args.lora_adapter_path)


    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.multi_modal_projector.load_state_dict(checkpoint['projector_state_dict'])
    model.code_encoder.position_embedding.load_state_dict(checkpoint['pos_embedding_state_dict'])
    model.eval() 
    print("Checkpoint loaded successfully.")

    # --- 3. Load Evaluation Dataset and Loop ---
    print(f"Loading evaluation data from: {args.eval_dataset_path}")
    with open(args.eval_dataset_path, "r") as f:
        eval_dataset = json.load(f)

    results = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for item in tqdm(eval_dataset, desc="Evaluating CodePatch"):
        code_string = item["code"]
        reference_description = item["description"]

        # --- Prepare Inputs for this item ---
        code_summary = ' '.join(code_string.splitlines()[:3])
        prompt_text = f"The MATLAB code is: {code_summary}... Describe the plot generated by this code, including shape, min/max, and key features:"
        
        original_inputs = processor(code=code_string, text=prompt_text)
        
        num_patches = original_inputs['code_input_ids'].shape[1]
        prompt_tokens = original_inputs['prompt_input_ids'].shape[1]
        total_input_tokens = num_patches + prompt_tokens
        initial_kv_cache_tokens = prompt_tokens

        # --- Calculate actual KV cache size ---
        # For CodePatch, only the text prompt fills the KV cache initially.
        kv_cache_size_bytes = 2 * num_layers * hidden_size * initial_kv_cache_tokens * bytes_per_param
        kv_cache_size_mb = round(kv_cache_size_bytes / (1024**2), 2)

        inputs = {k: v.clone() for k, v in original_inputs.items()}
        
        # --- Generate Text for this item ---
        generated_token_ids = []
        generated_text_so_far = ""
        unrelated_keywords = ['sine', 'cosine', 'random', 'python', 'optimizer']
        with torch.no_grad():
            for i in range(args.max_new_tokens):
                outputs = model(
                    code_input_ids=inputs['code_input_ids'],
                    code_attention_mask=inputs['code_attention_mask'],
                    prompt_input_ids=inputs['prompt_input_ids'],
                    prompt_attention_mask=inputs['prompt_attention_mask'],
                    debug=args.debug,
                )
                logits = outputs["logits"]
                next_token_logits = logits[:, -1, :]
                next_token_logits = torch.clamp(next_token_logits, min=-30, max=30)

                if generated_token_ids:
                    for token_id in set(generated_token_ids):
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= args.repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= args.repetition_penalty

                next_token_logits = next_token_logits / args.temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = sample_top_p(probs, args.top_p)
                
                if next_token_id.item() == processor.text_tokenizer.eos_token_id or i >= 150:
                    break
                
                generated_token = processor.text_tokenizer.decode(next_token_id.item(), skip_special_tokens=True)
                generated_text_so_far += generated_token

                if i > 50 and generated_text_so_far[-50:] == generated_text_so_far[-100:-50]:
                    break
                if i > 30 and any(kw in generated_text_so_far.lower() for kw in unrelated_keywords if kw not in code_string.lower()):
                    break
                try:
                    if len(generated_text_so_far) > 30 and detect(generated_text_so_far) != 'en':
                        break
                except LangDetectException:
                    pass

                generated_token_ids.append(next_token_id.item())
                
                inputs["prompt_input_ids"] = torch.cat([inputs["prompt_input_ids"], next_token_id], dim=1)
                inputs["prompt_attention_mask"] = torch.cat(
                    [inputs["prompt_attention_mask"], torch.ones_like(next_token_id)], dim=1
                )

        generated_text = processor.text_tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        scores = scorer.score(reference_description, generated_text)

        results.append({
            "code": code_string,
            "reference_description": reference_description,
            "generated_description": generated_text,
            "metrics": {
                "input_patches": num_patches,
                "input_prompt_tokens": prompt_tokens,
                "total_input_constructs": total_input_tokens,
                "initial_kv_cache_tokens": initial_kv_cache_tokens,
                "kv_cache_size_mb": kv_cache_size_mb,
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        })

    # --- 4. Save Final Results ---
    print(f"\nSaving evaluation results to: {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    main()
