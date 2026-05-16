"""Negative Binomial output head and NLL loss."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NBHead(nn.Module):
    """Maps predicted log1p expression to NB distribution parameters (mu, theta).

    Architecture:
        shared: Linear(n_genes, H) -> SiLU -> Linear(H, H) -> SiLU
        mu_head:    Linear(H, n_genes)
        theta_head: Linear(H, n_genes)
    """

    def __init__(self, n_genes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, n_genes)
        self.theta_head = nn.Linear(hidden_dim, n_genes)

    def forward(self, x1_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1_hat: (B, n_genes) predicted (or true) perturbed log1p expression.

        Returns:
            mu:    (B, n_genes)  positive NB mean.
            theta: (B, n_genes)  positive NB dispersion.
        """
        h = self.shared(x1_hat)
        mu = F.softplus(self.mu_head(h)) + 1e-6
        theta = F.softplus(self.theta_head(h)) + 1e-6
        return mu, theta


def nb_nll(
    counts: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Negative Binomial negative log-likelihood, mean-reduced over all elements.

    Parameterisation:
        total_count = theta
        probs       = theta / (theta + mu)   (success probability)

    So E[X] = mu, Var[X] = mu + mu^2/theta.

    Args:
        counts: (B, n_genes) raw integer counts cast to float.
        mu:     (B, n_genes) NB mean.
        theta:  (B, n_genes) NB dispersion (total_count).

    Returns:
        Scalar mean NLL.
    """
    dist = torch.distributions.NegativeBinomial(
        total_count=theta,
        probs=theta / (theta + mu),
    )
    return -dist.log_prob(counts).mean()
