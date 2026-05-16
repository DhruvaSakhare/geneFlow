"""PerturbationFlowModel: flow matching + NB head, loss, and RK4 inference."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from perturbation_fm.model.flow_net import FlowNet
from perturbation_fm.model.nb_head import NBHead, nb_nll


class PerturbationFlowModel(nn.Module):
    """Combines FlowNet and NBHead into a single trainable module.

    Training uses:
        loss = loss_fm + lambda_nb * loss_nb
    where loss_fm trains the flow network and loss_nb trains only the NB head
    (they share no parameters).

    Inference runs RK4 ODE integration to transport x0 → x1, then samples
    integer counts from NB(mu, theta).
    """

    def __init__(
        self,
        n_genes: int,
        esm_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
        lambda_nb: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_nb = lambda_nb
        self.flow_net = FlowNet(n_genes, esm_dim, hidden_dim, num_layers, dropout)
        self.nb_head = NBHead(n_genes)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1_log1p: torch.Tensor,
        x1_counts: torch.Tensor,
        esm_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined flow-matching + NB loss.

        Args:
            x0:         (B, n_genes)  control cell log1p expression.
            x1_log1p:   (B, n_genes)  true perturbed log1p expression.
            x1_counts:  (B, n_genes)  true raw counts (float).
            esm_emb:    (B, esm_dim)  perturbation gene embedding.

        Returns:
            dict with keys "loss", "loss_fm", "loss_nb".
        """
        B = x0.shape[0]
        device = x0.device

        # 1. Minibatch optimal transport pairing.
        # Reorder x0 so each (x0, x1) pair minimises total ||x0 - x1||².
        # This straightens trajectories and reduces velocity field variance.
        with torch.no_grad():
            C = torch.cdist(x0, x1_log1p).cpu().numpy()          # (B, B)
            row_idx, col_idx = linear_sum_assignment(C)
            row_idx = torch.from_numpy(row_idx).to(device)
            col_idx = torch.from_numpy(col_idx).to(device)
        x0        = x0[row_idx]
        x1_log1p  = x1_log1p[col_idx]
        x1_counts = x1_counts[col_idx]
        esm_emb   = esm_emb[col_idx]

        # 2. Sample t ~ Uniform(0, 1)
        t = torch.rand(B, 1, device=device)

        # 3. Interpolate
        x_t = (1.0 - t) * x0 + t * x1_log1p

        # 4. True velocity
        u_target = x1_log1p - x0

        # 5. Predicted velocity
        u_pred = self.flow_net(x_t, t, esm_emb)

        # 6. Flow matching MSE
        loss_fm = F.mse_loss(u_pred, u_target)

        # 7–8. NB head trained on TRUE x1 (not predicted)
        mu, theta = self.nb_head(x1_log1p)
        loss_nb = nb_nll(x1_counts, mu, theta)

        # 9. Combined loss
        loss = loss_fm + self.lambda_nb * loss_nb

        return {
            "loss": loss,
            "loss_fm": loss_fm.item(),
            "loss_nb": loss_nb.item(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x0_log1p: torch.Tensor,
        esm_emb: torch.Tensor,
        steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """Transport x0 to predicted x1 via RK4 ODE integration.

        Args:
            x0_log1p: (B, n_genes)  control cell log1p expression.
            esm_emb:  (esm_dim,) or (B, esm_dim)  gene embedding.
            steps:    number of RK4 integration steps.

        Returns:
            dict with keys "x1_log1p", "mu", "theta", "counts".
        """
        device = x0_log1p.device
        B = x0_log1p.shape[0]

        # Broadcast single embedding to batch dimension.
        if esm_emb.dim() == 1:
            esm_emb = esm_emb.unsqueeze(0).expand(B, -1)

        x = x0_log1p.clone().float()
        dt = 1.0 / steps

        def v(x_: torch.Tensor, t_val: float) -> torch.Tensor:
            t_tensor = torch.full((B, 1), t_val, device=device, dtype=torch.float32)
            return self.flow_net(x_, t_tensor, esm_emb)

        for i in range(steps):
            t_i = i * dt
            k1 = v(x,                   t_i)
            k2 = v(x + k1 * (dt / 2),  t_i + dt / 2)
            k3 = v(x + k2 * (dt / 2),  t_i + dt / 2)
            k4 = v(x + k3 * dt,         t_i + dt)
            x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)

        x1_hat = x
        mu, theta = self.nb_head(x1_hat)

        nb_dist = torch.distributions.NegativeBinomial(
            total_count=theta,
            probs=theta / (theta + mu),
        )
        counts = nb_dist.sample()

        return {
            "x1_log1p": x1_hat,
            "mu": mu,
            "theta": theta,
            "counts": counts,
        }
