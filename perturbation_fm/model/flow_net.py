"""Vector field network v_theta(x_t, t, c) for flow matching."""

from __future__ import annotations

import torch
import torch.nn as nn

from perturbation_fm.model.blocks import AdaLN, ResBlock
from perturbation_fm.model.time_embedding import SinusoidalTimeEmbedding


class FlowNet(nn.Module):
    """Predicts the velocity field given interpolated state, time, and gene conditioning.

    Architecture:
        gene_proj  :  Linear(n_genes, H)
        time_emb   :  SinusoidalTimeEmbedding(H)
        esm_proj   :  Linear(E, H) -> SiLU -> Linear(H, H)
        cond       :  cat([time_emb(t), esm_proj(c)], dim=-1)  shape (B, 2H)
        8× ResBlock(H, 2H)
        AdaLN(H, 2H)
        out_head   :  Linear(H, n_genes)   — zero-initialized
    """

    def __init__(
        self,
        n_genes: int,
        esm_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        H = hidden_dim
        cond_dim = H * 2

        self.gene_proj = nn.Linear(n_genes, H)

        self.time_emb = SinusoidalTimeEmbedding(H)

        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        self.blocks = nn.ModuleList(
            [ResBlock(H, cond_dim, dropout) for _ in range(num_layers)]
        )

        self.final_adaLN = AdaLN(H, cond_dim)

        self.out_head = nn.Linear(H, n_genes)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, n_genes)  interpolated expression at time t.
            t:   (B, 1)        time in [0, 1].
            c:   (B, esm_dim)  gene embedding.

        Returns:
            velocity: (B, n_genes)
        """
        h = self.gene_proj(x_t)                              # (B, H)
        t_emb = self.time_emb(t)                             # (B, H)
        c_emb = self.esm_proj(c)                             # (B, H)
        cond = torch.cat([t_emb, c_emb], dim=-1)            # (B, 2H)

        for block in self.blocks:
            h = block(h, cond)

        h = self.final_adaLN(h, cond)
        return self.out_head(h)                              # (B, n_genes)
