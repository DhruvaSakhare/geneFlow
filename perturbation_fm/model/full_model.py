"""PerturbationFlowModel: flow matching + NB head, loss, and RK4 inference."""

from __future__ import annotations

from typing import Dict

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
        lambda_lfc: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_nb = lambda_nb
        self.lambda_lfc = lambda_lfc
        self.flow_net = FlowNet(n_genes, esm_dim, hidden_dim, num_layers, dropout)
        self.nb_head = NBHead(n_genes)
        # baseline_lfc: mean LFC across all training perturbations.
        # The flow learns the residual on top of this — model only has to predict
        # the perturbation-specific correction, not the shared stress response.
        self.register_buffer("baseline_lfc", torch.zeros(n_genes))

    def set_baseline_lfc(self, baseline_lfc: torch.Tensor) -> None:
        """Set the baseline LFC vector (called once after preprocessing)."""
        self.baseline_lfc.copy_(baseline_lfc.float())

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

        # 0. Residual prediction: subtract baseline LFC from FM target.
        # Model learns to predict deviation from average training perturbation,
        # not the full transport. baseline_lfc is added back during inference.
        # NB head keeps original x1_log1p for count distribution.
        x1_log1p_orig = x1_log1p
        x1_log1p = x1_log1p - self.baseline_lfc

        # 1. Per-perturbation OT pairing.
        # Cells with identical esm_emb belong to the same perturbation.
        # OT within each group straightens trajectories without mixing
        # control cells across perturbations.
        with torch.no_grad():
            _, inverse_idx = torch.unique(esm_emb, dim=0, return_inverse=True)
            x1_log1p_ot  = x1_log1p.clone()
            x1_counts_ot = x1_counts.clone()
            for p_idx in inverse_idx.unique():
                group = torch.where(inverse_idx == p_idx)[0]
                if len(group) <= 1:
                    continue
                C = torch.cdist(x0[group], x1_log1p[group]).cpu().numpy()
                _, col = linear_sum_assignment(C)
                col = torch.tensor(col, device=device)
                x1_log1p_ot[group]  = x1_log1p[group][col]
                x1_counts_ot[group] = x1_counts[group][col]
        x1_log1p  = x1_log1p_ot
        x1_counts = x1_counts_ot

        # 2. Sample t ~ Uniform(0, 1)
        t = torch.rand(B, 1, device=device)

        # 3. Interpolate
        x_t = (1.0 - t) * x0 + t * x1_log1p

        # 4. True velocity
        u_target = x1_log1p - x0

        # 4. Predicted velocity
        u_pred = self.flow_net(x_t, t, esm_emb)

        # 5. Gene-activity-weighted FM loss.
        # ~96% of genes are non-DE (near-zero velocity) and drown out the DE
        # gene signal at a 23:1 ratio with uniform MSE. Weighting by mean |u|
        # per gene focuses gradients on genes that actually change.
        with torch.no_grad():
            weights = u_target.abs().mean(dim=0) + 1e-6   # (n_genes,)
            weights = weights / weights.mean()             # mean weight = 1
        loss_fm = (F.mse_loss(u_pred, u_target, reduction="none") * weights).mean()

        # 6. Per-perturbation mean-LFC loss.
        # FM loss penalizes individual velocity errors. This penalizes the mean
        # velocity per perturbation group — directly optimizing what Pearson r
        # measures. Uses already-computed u_pred so no extra forward pass.
        lfc_losses = []
        for p_idx in inverse_idx.unique():
            group = torch.where(inverse_idx == p_idx)[0]
            if len(group) < 2:
                continue
            pred_mean = u_pred[group].mean(dim=0)
            true_mean = u_target[group].mean(dim=0)
            lfc_losses.append(F.mse_loss(pred_mean, true_mean))
        loss_lfc = torch.stack(lfc_losses).mean() if lfc_losses else torch.tensor(0.0, device=device)

        # 7–8. NB head trained on original (non-residual) x1
        mu, theta = self.nb_head(x1_log1p_orig)
        loss_nb = nb_nll(x1_counts, mu, theta)

        # 9. Combined loss
        loss = loss_fm + self.lambda_lfc * loss_lfc + self.lambda_nb * loss_nb

        return {
            "loss": loss,
            "loss_fm": loss_fm.item(),
            "loss_lfc": loss_lfc.item(),
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

        # Add baseline LFC back: ODE produced the residual prediction.
        x1_hat = x + self.baseline_lfc
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
