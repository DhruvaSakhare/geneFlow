"""Two UMAP figures on the held-out test set.

Figure 1 — trajectory UMAP (4 panels, t=0 / 0.25 / 0.75 / 1.0):
    Control cells are grey in every panel; predicted cells are coloured
    by perturbation.

Figure 2 — predicted vs true comparison (2 panels, side-by-side):
    Left:  control (grey) + model predictions at t=1.0 (coloured)
    Right: control (grey) + true perturbed cells from test set (coloured)
    Both panels share the same UMAP embedding (fit once on all three groups).

Usage:
    python -m perturbation_fm.evaluation.plot_umap_test \\
        --config perturbation_fm/configs/default.yaml \\
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt \\
        --test_path /work3/s225191/geneFlow/data/test.h5ad \\
        --out_traj /work3/s225191/geneFlow/figures/umap_trajectory_test.png \\
        --out_compare /work3/s225191/geneFlow/figures/umap_compare_test.png
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
    """RK4 ODE integration, saving cell states at requested time fractions."""
    device = x0.device
    B = x0.shape[0]
    if esm_emb.dim() == 1:
        esm_emb = esm_emb.unsqueeze(0).expand(B, -1)

    save_at = {round(t * n_steps): t for t in times}
    x = x0.clone().float()
    dt = 1.0 / n_steps
    saved: dict[float, torch.Tensor] = {}

    if 0 in save_at:
        saved[save_at[0]] = x.clone()

    def v(x_, t_val):
        t_tensor = torch.full((B, 1), t_val, device=device, dtype=torch.float32)
        return model.flow_net(x_, t_tensor, esm_emb)

    for i in range(n_steps):
        t_i = i * dt
        k1 = v(x,                  t_i)
        k2 = v(x + k1 * (dt / 2),  t_i + dt / 2)
        k3 = v(x + k2 * (dt / 2),  t_i + dt / 2)
        k4 = v(x + k3 * dt,        t_i + dt)
        x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)
        if (i + 1) in save_at:
            saved[save_at[i + 1]] = x.clone()

    return saved


def fit_umap(arrays: list[np.ndarray], n_pca: int = 50) -> tuple[UMAP, PCA, np.ndarray]:
    X = np.concatenate(arrays, axis=0)
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    X_umap = reducer.fit_transform(X_pca)
    return reducer, pca, X_umap


def make_color_map(perts: list[str]) -> dict[str, np.ndarray]:
    colors = plt.cm.tab20(np.linspace(0, 1, len(perts)))
    return {p: colors[i] for i, p in enumerate(perts)}


def set_ax_style(ax, title: str, xmin, xmax, ymin, ymax):
    pad_x, pad_y = 0.05 * (xmax - xmin), 0.05 * (ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--test_path",   required=True)
    parser.add_argument("--out_traj",    default="umap_trajectory_test.png")
    parser.add_argument("--out_compare", default="umap_compare_test.png")
    parser.add_argument("--n_seeds",     type=int, default=100,
                        help="Control cells used as ODE seeds per perturbation")
    parser.add_argument("--n_true",      type=int, default=100,
                        help="True perturbed cells sampled per perturbation for comparison")
    parser.add_argument("--ode_steps",   type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Pass test_path in the val slot — same gene selection, correctly held out.
    data_dict = preprocess(cfg["train_path"], args.test_path)
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
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    test_log1p  = data_dict["val_log1p"]
    test_genes  = np.asarray(data_dict["val_genes"])
    test_ctrl   = data_dict["val_ctrl_mask"]
    perts       = sorted(np.unique(test_genes[~test_ctrl]).tolist())
    color_map   = make_color_map(perts)
    print(f"Test perturbations ({len(perts)}): {perts}")

    rng = np.random.default_rng(42)

    # --- Sample control cells as ODE seeds ---
    ctrl_idx = np.where(test_ctrl)[0]
    ctrl_idx = rng.choice(ctrl_idx, min(args.n_seeds, len(ctrl_idx)), replace=False)
    ctrl_cells = test_log1p[ctrl_idx]
    seed_t = torch.from_numpy(ctrl_cells).float().to(device)

    # --- Integrate ODE for every test perturbation ---
    pred: dict[str, dict[float, np.ndarray]] = {}
    true: dict[str, np.ndarray] = {}
    for pert in perts:
        esm_emb = gene_emb_map.get(pert, torch.zeros(cfg["esm_dim"])).to(device)
        states = integrate_at_times(model, seed_t, esm_emb,
                                    times=TIME_POINTS, n_steps=args.ode_steps)
        pred[pert] = {t: states[t].cpu().numpy() for t in states}

        # Sample true perturbed cells for this perturbation
        pert_idx = np.where((~test_ctrl) & (test_genes == pert))[0]
        chosen = rng.choice(pert_idx, min(args.n_true, len(pert_idx)), replace=False)
        true[pert] = test_log1p[chosen]

    # ==========================================================================
    # Figure 1 — Trajectory UMAP
    # ==========================================================================
    print("\nFitting UMAP for trajectory figure...")
    blocks_traj = {"control": ctrl_cells}
    for pert in perts:
        for t in TIME_POINTS:
            if t > 0.0:
                blocks_traj[f"{pert}__t{t}"] = pred[pert][t]

    names_traj = list(blocks_traj.keys())
    sizes_traj = [blocks_traj[n].shape[0] for n in names_traj]
    _, _, X_umap_traj = fit_umap([blocks_traj[n] for n in names_traj])

    offsets = np.cumsum([0] + sizes_traj)
    slices_traj = {n: X_umap_traj[offsets[i]:offsets[i + 1]]
                   for i, n in enumerate(names_traj)}

    ctrl_xy = slices_traj["control"]
    xmin, xmax = X_umap_traj[:, 0].min(), X_umap_traj[:, 0].max()
    ymin, ymax = X_umap_traj[:, 1].min(), X_umap_traj[:, 1].max()

    fig1, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for ax, t in zip(axes, TIME_POINTS):
        ax.scatter(ctrl_xy[:, 0], ctrl_xy[:, 1],
                   c="lightgrey", s=12, alpha=0.6, linewidths=0, zorder=1)
        if t > 0.0:
            for pert in perts:
                xy = slices_traj[f"{pert}__t{t}"]
                ax.scatter(xy[:, 0], xy[:, 1],
                           c=[color_map[pert]], s=14, alpha=0.7,
                           linewidths=0, zorder=2)
        set_ax_style(ax, f"t = {t}", xmin, xmax, ymin, ymax)

    legend_handles = [plt.scatter([], [], c="lightgrey", s=30, label="Control (t=0)")]
    for p in perts:
        legend_handles.append(plt.scatter([], [], c=[color_map[p]], s=30, label=p))
    fig1.legend(handles=legend_handles, title="Perturbation",
                loc="center right", bbox_to_anchor=(1.0, 0.5),
                fontsize=8, frameon=True)
    fig1.suptitle("Predicted cell-state trajectories on test perturbations",
                  fontsize=14, y=1.0)
    fig1.tight_layout(rect=(0, 0, 0.88, 0.98))

    Path(args.out_traj).parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(args.out_traj, dpi=150, bbox_inches="tight")
    print(f"Saved trajectory UMAP: {args.out_traj}")

    # ==========================================================================
    # Figure 2 — Predicted vs True comparison at t=1.0
    # ==========================================================================
    print("\nFitting UMAP for comparison figure...")
    pred_final = {p: pred[p][1.0] for p in perts}   # predicted at t=1
    true_final  = {p: true[p]     for p in perts}    # true perturbed cells

    blocks_cmp = {"control": ctrl_cells}
    for p in perts:
        blocks_cmp[f"pred__{p}"] = pred_final[p]
        blocks_cmp[f"true__{p}"] = true_final[p]

    names_cmp = list(blocks_cmp.keys())
    sizes_cmp = [blocks_cmp[n].shape[0] for n in names_cmp]
    _, _, X_umap_cmp = fit_umap([blocks_cmp[n] for n in names_cmp])

    offsets_cmp = np.cumsum([0] + sizes_cmp)
    slices_cmp = {n: X_umap_cmp[offsets_cmp[i]:offsets_cmp[i + 1]]
                  for i, n in enumerate(names_cmp)}

    ctrl_xy2 = slices_cmp["control"]
    xmin2, xmax2 = X_umap_cmp[:, 0].min(), X_umap_cmp[:, 0].max()
    ymin2, ymax2 = X_umap_cmp[:, 1].min(), X_umap_cmp[:, 1].max()

    fig2, (ax_pred, ax_true) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, prefix, title in [
        (ax_pred, "pred__", "Model predictions (t=1.0)"),
        (ax_true, "true__", "True perturbed cells"),
    ]:
        ax.scatter(ctrl_xy2[:, 0], ctrl_xy2[:, 1],
                   c="lightgrey", s=12, alpha=0.5, linewidths=0, zorder=1,
                   label="Control")
        for p in perts:
            xy = slices_cmp[f"{prefix}{p}"]
            marker = "o" if prefix == "pred__" else "^"
            ax.scatter(xy[:, 0], xy[:, 1],
                       c=[color_map[p]], s=16, alpha=0.7,
                       marker=marker, linewidths=0, zorder=2)
        set_ax_style(ax, title, xmin2, xmax2, ymin2, ymax2)

    legend_handles2 = [plt.scatter([], [], c="lightgrey", s=30, label="Control")]
    for p in perts:
        legend_handles2.append(
            plt.scatter([], [], c=[color_map[p]], s=30, label=p))
    legend_handles2 += [
        plt.scatter([], [], c="grey", marker="o", s=30, label="Predicted (●)"),
        plt.scatter([], [], c="grey", marker="^", s=30, label="True (▲)"),
    ]
    fig2.legend(handles=legend_handles2, title="Perturbation",
                loc="center right", bbox_to_anchor=(1.0, 0.5),
                fontsize=8, frameon=True)
    fig2.suptitle("Predicted vs True cell states at t=1.0 (test perturbations)",
                  fontsize=14)
    fig2.tight_layout(rect=(0, 0, 0.88, 1.0))

    Path(args.out_compare).parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.out_compare, dpi=150, bbox_inches="tight")
    print(f"Saved comparison UMAP: {args.out_compare}")


if __name__ == "__main__":
    main()
