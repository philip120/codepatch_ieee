# shared/projector.py
"""
Projector (Bottleneck MLP)

Projects patch embeddings to Qwen's embedding space.

Architecture: 3072 → 512 → 1536
- Bottleneck forces compression
- Heavy regularization for small datasets
"""
import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Bottleneck MLP: patch_dim → bottleneck → qwen_dim

    Architecture:
        Linear(in → bottleneck)
        → LayerNorm
        → GELU
        → Dropout
        → Linear(bottleneck → out)
        → LayerNorm(out)
        → learned scalar (output_scale)

    The final LayerNorm normalises direction (unit-variance per token),
    preventing feature collapse and keeping gradients stable. A single
    learned scalar `output_scale` then controls the magnitude. It is
    initialised to 0.022 ≈ 1.09 / 50, placing the initial projected norm
    near Qwen3-4B's token embedding norm (~1.09). During training the
    scalar adapts freely via gradient descent.

    This avoids two failure modes:
      - LayerNorm alone: norm fixed at sqrt(out_dim) ≈ 50 (45× too large,
        cannot adapt)
      - No LayerNorm: norm drifts upward during training (optimizer exploits
        large norms to dominate the residual stream)
    """

    # Qwen3-4B token embedding norm (measured: mean=1.09, std=0.17)
    QWEN_TOKEN_NORM = 1.09

    def __init__(
        self,
        in_dim: int = 3072,        # patch_size * 768
        bottleneck_dim: int = 512,  # compression layer
        out_dim: int = 2560,        # Qwen embedding dim
        dropout: float = 0.4,       # aggressive for small data
    ):
        super().__init__()

        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            # Compress
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Expand to Qwen space — no final LayerNorm.
            # A final LayerNorm pins norm to sqrt(out_dim) ≈ 50 (cannot adapt).
            # A learned scalar output_scale to compensate damps projector gradients
            # by 45× (output_scale ≈ 0.022), causing representation collapse
            # (proj_var → 0). The linear layer learns the correct output scale
            # directly via gradient descent with weight_decay keeping it bounded.
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project patch embeddings to Qwen space.

        Args:
            x: [num_patches, in_dim] patch embeddings

        Returns:
            [num_patches, out_dim] projected embeddings
        """
        return self.net(x)

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
