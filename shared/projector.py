# shared/projector.py
"""
Projector (Single Linear)

Projects patch embeddings to Qwen's embedding space.

Architecture: patch_size*768 → qwen_dim  (single linear, Kaiming init)
- No bottleneck, no LayerNorm, no norm drift issues
- Kaiming init produces output norms close to Qwen token norms (~1.09)
- Gradients flow directly without dampening from learned scalars
"""
import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Single linear projection: patch_dim → qwen_dim

    A bottleneck MLP with a final LayerNorm pins the output norm to
    sqrt(out_dim) ≈ 50, which is 45× larger than Qwen token norms (~1.09),
    causing the projected embeddings to dominate the residual stream and
    bypass the transformer layers. Compensating with a learned scalar damps
    projector gradients by the same 45× factor, collapsing representations.

    A single Linear layer avoids both failure modes:
      - Kaiming init: fan_in=in_dim, so output std ≈ sqrt(2/in_dim) * input_std
        which for unit-normal inputs gives norm ≈ sqrt(out_dim * 2/in_dim)
        ≈ sqrt(2560 * 2/3072) ≈ 1.29 — close to Qwen's ~1.09.
      - No LayerNorm means norm can adapt freely during training.
      - No bottleneck means gradients are not compressed through a narrow layer.
    """

    def __init__(
        self,
        in_dim: int = 3072,        # patch_size * 768
        bottleneck_dim: int = 512,  # unused, kept for API compatibility
        out_dim: int = 2560,        # Qwen embedding dim
        dropout: float = 0.0,       # unused, kept for API compatibility
    ):
        super().__init__()

        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = out_dim

        self.net = nn.Linear(in_dim, out_dim)

    # Max-norm ceiling for projected tokens.
    # Keeps output norms in a range Qwen can use (Qwen token norm ≈ 1.09).
    # Set conservatively at 3.0 — acts only when norms start to explode.
    MAX_NORM = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project patch embeddings to Qwen space.

        Args:
            x: [num_patches, in_dim] patch embeddings

        Returns:
            [num_patches, out_dim] projected embeddings
        """
        out = self.net(x)
        # Max-norm clamp: scale DOWN tokens exceeding MAX_NORM, never scale up.
        # Below the ceiling, scale=1.0 so gradients are completely unaffected.
        # Only when a token's norm is actively growing past the ceiling does
        # this engage — preventing runaway norm drift without dampening learning.
        norms = out.norm(dim=-1, keepdim=True)
        scale = (self.MAX_NORM / norms).clamp(max=1.0)
        return out * scale

    def num_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    PATCH_SIZE = 4
    in_dim = PATCH_SIZE * 768  # 3072
    bottleneck = 512
    out_dim = 1536

    # Fake patch embeddings
    num_patches = 3
    patch_emb = torch.randn(num_patches, in_dim, device=device)

    projector = Projector(
        in_dim=in_dim,
        bottleneck_dim=bottleneck,
        out_dim=out_dim,
        dropout=0.4
    ).to(device)

    output = projector(patch_emb)

    print("Projector Test")
    print("=" * 60)
    print(f"\n  Architecture: {in_dim} → {bottleneck} → {out_dim}")
    print(f"\n  Input:  {patch_emb.shape}")
    print(f"  Output: {output.shape}")
    print(f"\n  Trainable params: {projector.num_parameters():,}")
    print(f"\n  Breakdown:")
    print(f"    Linear 1: {in_dim} × {bottleneck} = {in_dim * bottleneck:,}")
    print(f"    Linear 2: {bottleneck} × {out_dim} = {bottleneck * out_dim:,}")
