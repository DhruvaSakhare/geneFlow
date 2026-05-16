"""AdaLN and ResBlock used throughout the flow network."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive Layer Norm — conditions a hidden state on an external signal.

    Applies LayerNorm(x) then scales/shifts by learned functions of `cond`.
    The conditioning projection is zero-initialized so the block acts as
    plain LayerNorm (no conditioning effect) at the start of training.
    """

    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        # Zero-init: scale starts at 0 → (1+scale)*LN(x) = LN(x) at init.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, hidden_dim)
            cond: (B, cond_dim)

        Returns:
            (B, hidden_dim)
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class ResBlock(nn.Module):
    """Residual block conditioned via AdaLN.

    Structure:  x + MLP(AdaLN(x, cond))
    """

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.adaLN = AdaLN(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, hidden_dim)
            cond: (B, cond_dim)

        Returns:
            (B, hidden_dim)
        """
        return x + self.mlp(self.adaLN(x, cond))
