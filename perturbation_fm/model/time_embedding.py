"""Sinusoidal time embedding with MLP projection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for scalar time values in [0, 1].

    Follows the DDPM schedule: frequencies are log-spaced powers of 10000.
    The raw sinusoidal features are projected through a 2-layer MLP with SiLU.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding."
        self.dim = dim
        half_dim = dim // 2

        # Frequencies: exp(-log(10000) * i / (half_dim - 1))  for i = 0…half_dim-1
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)
        )
        self.register_buffer("freqs", freqs)  # (half_dim,)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1)  time values in [0, 1].

        Returns:
            (B, dim) embedding tensor.
        """
        t = t.reshape(-1, 1).float()                         # (B, 1)
        args = t * self.freqs.unsqueeze(0)                   # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)
