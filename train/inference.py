# train/inference.py
"""
Code-to-Thought Inference Pipeline

Takes MATLAB code → Semantic Nodes → CodeBERT → MLP Projection → Qwen → Pseudocode
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from train.semantic_adapter import code_to_nodes
from train.model import codebert_embed_nodes, ProjectionMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache
_qwen_tokenizer = None
_qwen_model = None
_mlp = None


def load_models(checkpoint_path: str = "mlp_best.pt"):
    """Load all models for inference."""
    global _qwen_tokenizer, _qwen_model, _mlp

    # Load MLP
    if _mlp is None:
        print("Loading projection MLP...")
        _mlp = ProjectionMLP(in_dim=768, out_dim=1536).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        _mlp.load_state_dict(checkpoint['model_state_dict'])
        _mlp.eval()

    # Load Qwen for generation
    if _qwen_model is None:
        print("Loading Qwen2 (1.5B) for generation...")
        model_name = "Qwen/Qwen2-1.5B"
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        _qwen_model.to(DEVICE)
        _qwen_model.eval()

    return _mlp, _qwen_tokenizer, _qwen_model


def generate_pseudocode(
    matlab_code: str,
    checkpoint_path: str = "mlp_best.pt",
    max_new_tokens: int = 256,
    num_thought_tokens: int = 8,
    temperature: float = 0.7,
    show_nodes: bool = False,
):
    """
    Generate pseudocode from MATLAB code using the C2T pipeline.

    Args:
        matlab_code: Input MATLAB code
        checkpoint_path: Path to trained MLP checkpoint
        max_new_tokens: Maximum tokens to generate
        num_thought_tokens: Number of "thought" embeddings to inject
        temperature: Sampling temperature
        show_nodes: Print extracted semantic nodes

    Returns:
        Generated pseudocode string
    """
    mlp, tokenizer, model = load_models(checkpoint_path)

    # 1. Extract semantic nodes from MATLAB code
    nodes = code_to_nodes(matlab_code)
    if not nodes:
        return "Error: Could not parse MATLAB code"

    if show_nodes:
        print(f"\nExtracted {len(nodes)} semantic nodes:")
        for i, n in enumerate(nodes):
            print(f"  [{i}] {n}")
        print()

    # 2. Embed code nodes with CodeBERT
    with torch.no_grad():
        z_code = codebert_embed_nodes(nodes)  # [768]

        # 3. Project to Qwen space with MLP
        z_thought = mlp(z_code)  # [1536]

        # 4. Expand to multiple "thought tokens"
        # Repeat the thought embedding to create a sequence of thought tokens
        thought_embeds = z_thought.unsqueeze(0).unsqueeze(0)  # [1, 1, 1536]
        thought_embeds = thought_embeds.expand(1, num_thought_tokens, -1)  # [1, num_thoughts, 1536]

        # 5. Create the text prompt
        prompt = "Based on the code analysis above, here is a clear pseudocode description:\n"

        # Tokenize prompt
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_embeds = model.get_input_embeddings()(prompt_tokens.input_ids)  # [1, seq, 1536]

        # 6. Concatenate: [thought_embeds] + [prompt_embeds]
        # Convert thought_embeds to same dtype as prompt_embeds
        thought_embeds = thought_embeds.to(prompt_embeds.dtype)
        input_embeds = torch.cat([thought_embeds, prompt_embeds], dim=1)

        # 7. Generate with Qwen
        # Create attention mask for the combined input
        attn_mask = torch.ones(1, input_embeds.shape[1], device=DEVICE)

        outputs = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode output (skip the input length)
        generated_ids = outputs[0]
        pseudocode = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return pseudocode


def batch_generate(
    codes: list[str],
    checkpoint_path: str = "mlp_best.pt",
    **kwargs
) -> list[str]:
    """Generate pseudocode for multiple MATLAB code snippets."""
    results = []
    for i, code in enumerate(codes):
        print(f"Processing {i+1}/{len(codes)}...")
        result = generate_pseudocode(code, checkpoint_path, **kwargs)
        results.append(result)
    return results


def interactive_mode(checkpoint_path: str = "mlp_best.pt"):
    """Interactive pseudocode generation."""
    print("=" * 60)
    print("MATLAB to Pseudocode Generator (C2T Pipeline)")
    print("=" * 60)
    print("\nEnter MATLAB code (end with empty line), or 'quit' to exit.\n")

    load_models(checkpoint_path)

    while True:
        print("-" * 40)
        print("Enter MATLAB code:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.lower() == "quit":
                print("Goodbye!")
                return
            if line == "":
                break
            lines.append(line)

        if not lines:
            continue

        matlab_code = "\n".join(lines)
        print("\nGenerating pseudocode...")

        pseudocode = generate_pseudocode(
            matlab_code,
            checkpoint_path=checkpoint_path,
            show_nodes=True,
            max_new_tokens=256,
        )

        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(pseudocode)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudocode from MATLAB code")
    parser.add_argument("--checkpoint", type=str, default="mlp_best.pt")
    parser.add_argument("--code", type=str, default=None, help="MATLAB code string")
    parser.add_argument("--file", type=str, default=None, help="Path to MATLAB file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thoughts", type=int, default=8, help="Number of thought tokens")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.checkpoint)
    elif args.file:
        with open(args.file, "r") as f:
            code = f.read()
        print(f"Processing: {args.file}")
        result = generate_pseudocode(
            code,
            checkpoint_path=args.checkpoint,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            num_thought_tokens=args.thoughts,
            show_nodes=True,
        )
        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(result)
    elif args.code:
        result = generate_pseudocode(
            args.code,
            checkpoint_path=args.checkpoint,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            num_thought_tokens=args.thoughts,
            show_nodes=True,
        )
        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(result)
    else:
        # Demo with example code
        demo_code = """
function y = calc(x)
    y = 0;
    for i = 1:length(x)
        y = y + x(i);
    end
end
"""
        print("Demo: Generating pseudocode for sample function")
        result = generate_pseudocode(
            demo_code,
            checkpoint_path=args.checkpoint,
            show_nodes=True,
        )
        print("\n" + "=" * 40)
        print("GENERATED PSEUDOCODE:")
        print("=" * 40)
        print(result)
