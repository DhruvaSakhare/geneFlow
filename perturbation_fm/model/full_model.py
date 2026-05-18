"""PerturbationFlowModel: flow matching loss + RK4 inference."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from perturbation_fm.model.flow_net import FlowNet


class PerturbationFlowModel(nn.Module):
    """Conditional flow-matching model operating in log1p gene-expression space.

    Training uses:
        loss = loss_fm + lambda_lfc * loss_lfc
    Inference runs RK4 ODE integration to transport x0 -> x1.
    """

    def __init__(
        self,
        n_genes: int,
        esm_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
        lambda_lfc: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_lfc = lambda_lfc
        self.flow_net = FlowNet(n_genes, esm_dim, hidden_dim, num_layers, dropout)
        self.register_buffer("baseline_lfc", torch.zeros(n_genes))

    def set_baseline_lfc(self, baseline_lfc: torch.Tensor) -> None:
        self.baseline_lfc.copy_(baseline_lfc.float())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1_log1p: torch.Tensor,
        esm_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow-matching loss with mean-LFC auxiliary term."""
        B = x0.shape[0]
        device = x0.device

        # Subtract baseline so model learns the perturbation-specific residual.
        x1_log1p = x1_log1p - self.baseline_lfc

        # Per-perturbation OT pairing within each batch.
        with torch.no_grad():
            _, inverse_idx = torch.unique(esm_emb, dim=0, return_inverse=True)
            x1_log1p_ot = x1_log1p.clone()
            for p_idx in inverse_idx.unique():
                group = torch.where(inverse_idx == p_idx)[0]
                if len(group) <= 1:
                    continue
                C = torch.cdist(x0[group], x1_log1p[group]).cpu().numpy()
                _, col = linear_sum_assignment(C)
                col = torch.tensor(col, device=device)
                x1_log1p_ot[group] = x1_log1p[group][col]
        x1_log1p = x1_log1p_ot

        # Interpolate between control and perturbed.
        t = torch.rand(B, 1, device=device)
        x_t = (1.0 - t) * x0 + t * x1_log1p
        u_target = x1_log1p - x0
        u_pred = self.flow_net(x_t, t, esm_emb)

        # Gene-activity-weighted FM loss.
        with torch.no_grad():
            weights = u_target.abs().mean(dim=0) + 1e-6
            weights = weights / weights.mean()
        loss_fm = (F.mse_loss(u_pred, u_target, reduction="none") * weights).mean()

        # Per-perturbation mean-LFC loss.
        lfc_losses = []
        for p_idx in inverse_idx.unique():
            group = torch.where(inverse_idx == p_idx)[0]
            if len(group) < 2:
                continue
            pred_mean = u_pred[group].mean(dim=0)
            true_mean = u_target[group].mean(dim=0)
            lfc_losses.append(F.mse_loss(pred_mean, true_mean))
        loss_lfc = (
            torch.stack(lfc_losses).mean() if lfc_losses
            else torch.tensor(0.0, device=device)
        )

        loss = loss_fm + self.lambda_lfc * loss_lfc

        return {
            "loss": loss,
            "loss_fm": loss_fm.item(),
            "loss_lfc": loss_lfc.item(),
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
        """Transport x0 to predicted x1 via RK4 ODE integration."""
        device = x0_log1p.device
        B = x0_log1p.shape[0]

        if esm_emb.dim() == 1:
            esm_emb = esm_emb.unsqueeze(0).expand(B, -1)

        x = x0_log1p.clone().float()
        dt = 1.0 / steps

        def v(x_: torch.Tensor, t_val: float) -> torch.Tensor:
            t_tensor = torch.full((B, 1), t_val, device=device, dtype=torch.float32)
            return self.flow_net(x_, t_tensor, esm_emb)

        for i in range(steps):
            t_i = i * dt
            k1 = v(x,                  t_i)
            k2 = v(x + k1 * (dt / 2),  t_i + dt / 2)
            k3 = v(x + k2 * (dt / 2),  t_i + dt / 2)
            k4 = v(x + k3 * dt,        t_i + dt)
            x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)

        x1_hat = x + self.baseline_lfc
        return {"x1_log1p": x1_hat}
