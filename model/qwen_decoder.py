# model/qwen_decoder.py
"""
Step 6: Qwen Decoder

Frozen Qwen LLM that generates text from projected embeddings.

Training: [projected patches] + [target text] → loss
Inference: [projected patches] → generated text
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QwenDecoder:
    """
    Frozen Qwen decoder for text generation.

    Training mode:
        - Concatenates projected embeddings with target text
        - Computes next-token prediction loss (only on text tokens)

    Inference mode:
        - Takes projected embeddings as prompt
        - Generates text autoregressively
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B", device: str = None):
        self.device = device or DEVICE

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Qwen loaded on {self.device} (frozen)")

    def get_input_embeddings(self, text: str) -> torch.Tensor:
        """Get Qwen's embeddings for text tokens."""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)

        embeds = self.model.get_input_embeddings()(tokens.input_ids)
        return embeds, tokens

    def forward_train(
        self,
        projected: torch.Tensor,    # [num_patches, 1536]
        target_text: str,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            projected: [num_patches, 1536] from Projector
            target_text: ground truth pseudocode

        Returns:
            loss: cross-entropy loss on text tokens
        """
        # Add batch dimension: [num_patches, 1536] → [1, num_patches, 1536]
        projected = projected.unsqueeze(0)

        # Get target embeddings
        target_embeds, target_tokens = self.get_input_embeddings(target_text)

        # Match dtype (Qwen may use float16)
        projected = projected.to(target_embeds.dtype)

        # Concatenate: [patches] + [target text]
        # Shape: [1, num_patches + num_text_tokens, 1536]
        input_embeds = torch.cat([projected, target_embeds], dim=1)

        # Create attention mask
        num_patches = projected.shape[1]
        patch_mask = torch.ones(1, num_patches, device=self.device)
        attn_mask = torch.cat([patch_mask, target_tokens.attention_mask], dim=1)

        # Create labels
        # -100 = ignore (don't compute loss on patch tokens)
        # We only predict text tokens
        patch_labels = torch.full(
            (1, num_patches), -100, dtype=torch.long, device=self.device
        )
        text_labels = target_tokens.input_ids.clone()
        labels = torch.cat([patch_labels, text_labels], dim=1)

        # Forward through Qwen
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )

        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        projected: torch.Tensor,    # [num_patches, 1536]
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from projected embeddings.

        Args:
            projected: [num_patches, 1536] from Projector
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold

        Returns:
            generated text string
        """
        # Add batch dimension and match dtype
        projected = projected.unsqueeze(0).to(self.model.dtype)

        # Generate
        outputs = self.model.generate(
            inputs_embeds=projected,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Test
    print("QwenDecoder Test")
    print("=" * 60)

    decoder = QwenDecoder()

    # Fake projected embeddings (normally from Projector)
    num_patches = 3
    projected = torch.randn(num_patches, 1536, device=DEVICE)

    print(f"\n  Input projected: {projected.shape}")

    # Test training forward
    print("\n  Testing training forward...")
    target = "This function doubles the input if positive."
    loss = decoder.forward_train(projected, target)
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    print("\n  Testing generation...")
    output = decoder.generate(projected, max_new_tokens=32)
    print(f"  Generated: {output[:100]}...")
