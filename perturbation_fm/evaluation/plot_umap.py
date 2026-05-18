"""4-panel UMAP showing ODE evolution at t=0, 0.25, 0.75, 1.0.

Control cells (at t=0) are grey in every panel as a fixed reference.
For each panel t>0, we overlay the model's predicted cell states at that
time, colored by perturbation. All 15 val perturbations included.

Requires: pip install umap-learn scikit-learn

Usage:
    python -m perturbation_fm.evaluation.plot_umap \
        --config perturbation_fm/configs/default.yaml \
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt \
        --out /work3/s225191/geneFlow/figures/umap_time.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.model.full_model import PerturbationFlowModel


TIME_POINTS = [0.0, 0.25, 0.75, 1.0]


@torch.no_grad()
def integrate_at_times(
    model: PerturbationFlowModel,
    x0: torch.Tensor,
    esm_emb: torch.Tensor,
    times: list[float],
    n_steps: int = 50,
) -> dict[float, torch.Tensor]:
    """RK4 ODE, save state at requested fractions of total time. Returns dict
    keyed by time fraction → (B, n_genes) tensor with baseline_lfc added."""
    device = x0.device
    B = x0.shape[0]
    if esm_emb.dim() == 1:
        esm_emb = esm_emb.unsqueeze(0).expand(B, -1)

    # Which integer step corresponds to each requested time fraction.
    save_at = {round(t * n_steps): t for t in times}

    x = x0.clone().float()
    dt = 1.0 / n_steps
    saved: dict[float, torch.Tensor] = {}
    if 0 in save_at:
        saved[save_at[0]] = x.clone() + model.baseline_lfc

    def v(x_, t_val):
        t_tensor = torch.full((B, 1), t_val, device=device, dtype=torch.float32)
        return model.flow_net(x_, t_tensor, esm_emb)

    for i in range(n_steps):
        t_i = i * dt
        k1 = v(x,                   t_i)
        k2 = v(x + k1 * (dt / 2),  t_i + dt / 2)
        k3 = v(x + k2 * (dt / 2),  t_i + dt / 2)
        k4 = v(x + k3 * dt,         t_i + dt)
        x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)
        if (i + 1) in save_at:
            saved[save_at[i + 1]] = x.clone() + model.baseline_lfc

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--out",    default="umap_time.png")
    parser.add_argument("--n_seeds", type=int, default=100,
                        help="Control cells used as ODE seeds per perturbation")
    parser.add_argument("--ode_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    data_dict = preprocess(cfg["train_path"], cfg["val_path"])
    gene_emb_map = torch.load(cfg["emb_path"], map_location="cpu")

    model = PerturbationFlowModel(
        n_genes=data_dict["n_genes"],
        esm_dim=cfg["esm_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        lambda_lfc=cfg["lambda_lfc"],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Restore baseline_lfc
    train_log1p = data_dict["train_log1p"]
    train_genes_arr = np.asarray(data_dict["train_genes"])
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    ctrl_mean = data_dict["ctrl_mean_log1p"]
    _lfcs = [
        train_log1p[(~train_ctrl_mask) & (train_genes_arr == _p)].mean(axis=0) - ctrl_mean
        for _p in np.unique(train_genes_arr[~train_ctrl_mask])
    ]
    baseline_lfc_vec = np.mean(_lfcs, axis=0)
    model.set_baseline_lfc(torch.from_numpy(baseline_lfc_vec).to(device))

    val_log1p     = data_dict["val_log1p"]
    val_genes     = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]

    perts = sorted(np.unique(val_genes[~val_ctrl_mask]).tolist())
    print(f"Plotting {len(perts)} perturbations: {perts}")

    rng = np.random.default_rng(42)

    # Sample control cells (shared ODE seeds + background)
    ctrl_idx = np.where(val_ctrl_mask)[0]
    ctrl_idx = rng.choice(ctrl_idx, min(args.n_seeds, len(ctrl_idx)), replace=False)
    ctrl_cells = val_log1p[ctrl_idx]
    seed_t = torch.from_numpy(ctrl_cells).float().to(device)
    print(f"Control / seed cells: {len(ctrl_cells)}")

    # Integrate trajectories for every perturbation, save at requested times
    pred_by_pert_time: dict[str, dict[float, np.ndarray]] = {}
    for pert in perts:
        esm_emb = gene_emb_map.get(pert, torch.zeros(cfg["esm_dim"])).to(device)
        states = integrate_at_times(
            model, seed_t, esm_emb,
            times=TIME_POINTS, n_steps=args.ode_steps,
        )
        pred_by_pert_time[pert] = {t: states[t].cpu().numpy() for t in states}

    # ------------------------------------------------------------------
    # Stack everything for one joint UMAP fit
    # ------------------------------------------------------------------
    blocks = {"control": ctrl_cells}
    for pert in perts:
        for t in TIME_POINTS:
            if t == 0.0:
                continue                       # t=0 is the control cells themselves
            blocks[f"{pert}__t{t}"] = pred_by_pert_time[pert][t]

    names = list(blocks.keys())
    sizes = [blocks[n].shape[0] for n in names]
    X = np.concatenate([blocks[n] for n in names], axis=0)
    print(f"UMAP total points: {X.shape[0]}")

    print("PCA → 50 components...")
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X)

    print("UMAP → 2D...")
    X_umap = UMAP(
        n_components=2, random_state=42,
        n_neighbors=30, min_dist=0.3,
    ).fit_transform(X_pca)

    offsets = np.cumsum([0] + sizes)
    slices = {n: X_umap[offsets[i]:offsets[i + 1]] for i, n in enumerate(names)}

    # ------------------------------------------------------------------
    # Plot — 2×2 grid of UMAP panels
    # ------------------------------------------------------------------
    pert_colors = plt.cm.tab20(np.linspace(0, 1, len(perts)))
    color_map = {p: pert_colors[i] for i, p in enumerate(perts)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    ctrl_xy = slices["control"]

    # Shared axis limits so panels are comparable
    xmin, xmax = X_umap[:, 0].min(), X_umap[:, 0].max()
    ymin, ymax = X_umap[:, 1].min(), X_umap[:, 1].max()
    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)

    for ax, t in zip(axes, TIME_POINTS):
        # Control cells (grey) — in every panel
        ax.scatter(
            ctrl_xy[:, 0], ctrl_xy[:, 1],
            c="lightgrey", s=12, alpha=0.6,
            linewidths=0, zorder=1,
        )

        if t > 0.0:
            for pert in perts:
                xy = slices[f"{pert}__t{t}"]
                ax.scatter(
                    xy[:, 0], xy[:, 1],
                    c=[color_map[pert]], s=14, alpha=0.7,
                    linewidths=0, zorder=2,
                )

        ax.set_title(f"t = {t}", fontsize=14)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

    # Shared legend outside the grid
    legend_handles = [
        plt.scatter([], [], c="lightgrey", s=30, label="Control (t=0)")
    ]
    for p in perts:
        legend_handles.append(
            plt.scatter([], [], c=[color_map[p]], s=30, label=p)
        )
    fig.legend(
        handles=legend_handles, title="Perturbation",
        loc="center right", bbox_to_anchor=(1.0, 0.5),
        fontsize=9, frameon=True,
    )

    fig.suptitle(
        "Predicted cell-state evolution under CRISPRi perturbation\n"
        "(ODE integrated from control cells through gene expression space)",
        fontsize=15, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.98))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
