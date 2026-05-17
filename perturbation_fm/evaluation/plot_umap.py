"""UMAP visualization of cell-state trajectories under ODE integration.

Plots all 15 val perturbations: control cells (grey dots), true perturbed
cells (colored dots), model-predicted trajectories (colored lines with
start/end diamonds). Background scatter via scprep.

Requires: pip install umap-learn scikit-learn

Usage:
    python -m perturbation_fm.evaluation.plot_umap \
        --config perturbation_fm/configs/default.yaml \
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt \
        --out /work3/s225191/geneFlow/figures/umap.png
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


@torch.no_grad()
def integrate_trajectory(
    model: PerturbationFlowModel,
    x0: torch.Tensor,
    esm_emb: torch.Tensor,
    n_steps: int = 50,
    n_save: int = 20,
) -> torch.Tensor:
    """RK4 ODE with saved intermediate states. Returns (n_save+1, B, n_genes)."""
    device = x0.device
    B = x0.shape[0]
    if esm_emb.dim() == 1:
        esm_emb = esm_emb.unsqueeze(0).expand(B, -1)

    x = x0.clone().float()
    dt = 1.0 / n_steps
    save_steps = set(np.linspace(1, n_steps, n_save, dtype=int).tolist())
    saved = [x.clone() + model.baseline_lfc]

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
        if (i + 1) in save_steps:
            saved.append(x.clone() + model.baseline_lfc)

    return torch.stack(saved, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--out",    default="umap.png")
    parser.add_argument("--n_per_pert", type=int, default=100,
                        help="True cells sampled per perturbation")
    parser.add_argument("--n_traj", type=int, default=15,
                        help="Trajectories drawn per perturbation")
    parser.add_argument("--n_save", type=int, default=20,
                        help="Intermediate ODE points per trajectory")
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
        lambda_nb=cfg["lambda_nb"],
        lambda_lfc=cfg["lambda_lfc"],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Restore baseline_lfc from training data
    train_log1p = data_dict["train_log1p"]
    train_genes_arr = np.asarray(data_dict["train_genes"])
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    ctrl_mean = data_dict["ctrl_mean_log1p"]
    train_perts_unique = np.unique(train_genes_arr[~train_ctrl_mask])
    _lfcs = [
        train_log1p[(~train_ctrl_mask) & (train_genes_arr == _p)].mean(axis=0) - ctrl_mean
        for _p in train_perts_unique
    ]
    baseline_lfc_vec = np.mean(_lfcs, axis=0)
    model.set_baseline_lfc(torch.from_numpy(baseline_lfc_vec).to(device))

    val_log1p     = data_dict["val_log1p"]
    val_genes     = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]

    perts = sorted(np.unique(val_genes[~val_ctrl_mask]).tolist())
    print(f"Plotting {len(perts)} perturbations: {perts}")

    rng = np.random.default_rng(42)
    n_per = args.n_per_pert

    # ------------------------------------------------------------------
    # Sample cells (balanced control vs perturbed)
    # ------------------------------------------------------------------
    pert_cells = {}
    for pert in perts:
        idx = np.where((~val_ctrl_mask) & (val_genes == pert))[0]
        idx = rng.choice(idx, min(n_per, len(idx)), replace=False)
        pert_cells[pert] = val_log1p[idx]
    total_pert = sum(len(v) for v in pert_cells.values())

    ctrl_idx = np.where(val_ctrl_mask)[0]
    ctrl_idx = rng.choice(ctrl_idx, min(total_pert, len(ctrl_idx)), replace=False)
    ctrl_cells = val_log1p[ctrl_idx]
    print(f"Control: {len(ctrl_cells)}  | Perturbed total: {total_pert}")

    # Trajectory seed cells (subset of control)
    seed_idx = rng.choice(len(ctrl_cells), args.n_traj, replace=False)
    seed_t = torch.from_numpy(ctrl_cells[seed_idx]).float().to(device)

    # ------------------------------------------------------------------
    # Integrate trajectories per perturbation
    # ------------------------------------------------------------------
    traj_by_pert = {}
    for pert in perts:
        esm_emb = gene_emb_map.get(pert, torch.zeros(cfg["esm_dim"])).to(device)
        traj_by_pert[pert] = integrate_trajectory(
            model, seed_t, esm_emb,
            n_steps=args.ode_steps, n_save=args.n_save,
        ).cpu().numpy()
        print(f"  {pert}: traj shape {traj_by_pert[pert].shape}")

    # ------------------------------------------------------------------
    # Stack everything for joint UMAP fit
    # ------------------------------------------------------------------
    blocks = {"control": ctrl_cells}
    for pert in perts:
        blocks[f"{pert}__true"] = pert_cells[pert]
    for pert in perts:
        t = traj_by_pert[pert]
        blocks[f"{pert}__traj"] = t.reshape(-1, t.shape[-1])

    names = list(blocks.keys())
    sizes = [blocks[n].shape[0] for n in names]
    X = np.concatenate([blocks[n] for n in names], axis=0)
    print(f"Total UMAP points: {X.shape[0]}")

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
    # Plot: scprep background + trajectory overlay
    # ------------------------------------------------------------------
    # Build a stacked array of all background points (control + true) with labels.
    bg_xy = [slices["control"]]
    bg_labels = ["control"] * len(slices["control"])
    for pert in perts:
        bg_xy.append(slices[f"{pert}__true"])
        bg_labels += [pert] * len(slices[f"{pert}__true"])
    bg_xy = np.concatenate(bg_xy, axis=0)
    bg_labels = np.array(bg_labels)

    # Color map: control = grey, perturbations = hsv-spread
    pert_colors = plt.cm.tab20(np.linspace(0, 1, len(perts)))
    color_map = {p: pert_colors[i] for i, p in enumerate(perts)}
    color_map["control"] = (0.75, 0.75, 0.75, 1.0)

    fig, ax = plt.subplots(figsize=(13, 10))

    # Categorical scatter (control first so perturbed dots draw on top)
    legend_handles = []
    for label in ["control"] + perts:
        mask = bg_labels == label
        if not mask.any():
            continue
        color = color_map[label]
        ax.scatter(
            bg_xy[mask, 0], bg_xy[mask, 1],
            c=[color], s=8, alpha=0.6,
            linewidths=0, zorder=2 if label == "control" else 2.5,
        )
        legend_handles.append(
            plt.scatter([], [], c=[color], s=30, label=label)
        )
    bg_legend = ax.legend(
        handles=legend_handles, title="Perturbation",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=True,
    )
    ax.add_artist(bg_legend)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay trajectories
    for pert in perts:
        color = color_map[pert]
        traj_xy = slices[f"{pert}__traj"].reshape(args.n_save + 1, args.n_traj, 2)

        for i in range(args.n_traj):
            ax.plot(
                traj_xy[:, i, 0], traj_xy[:, i, 1],
                c=color, alpha=0.5, linewidth=0.7, zorder=3,
            )
        # End diamonds (per-pert color)
        ax.scatter(
            traj_xy[-1, :, 0], traj_xy[-1, :, 1],
            c=[color], s=55, marker="D",
            edgecolors="black", linewidths=0.4, zorder=4,
        )
    ax.set_title(
        "Predicted cell-state trajectories under CRISPRi perturbation\n"
        "(grey ● = control / ODE start, colored ◆ = predicted endpoint, lines = ODE path)"
    )
    ax.spines[["top", "right"]].set_visible(False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
