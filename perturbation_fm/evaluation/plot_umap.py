"""UMAP visualization of cell-state trajectories under ODE integration.

Stacks control cells, true perturbed cells, and the model's predicted
ODE trajectory (with intermediate time points) into one UMAP. Draws a
small number of trajectory lines per perturbation so the flow is visible.

Caveat: the model runs in 4721-dim log1p space, not in UMAP space.
Trajectory lines are the UMAP projection of the actual ODE path — they
are real points the cell passes through, but their curvature in 2D is a
UMAP artifact, not a property of the flow.

Usage:
    python -m perturbation_fm.evaluation.plot_umap \
        --config perturbation_fm/configs/default.yaml \
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt \
        --out /work3/s225191/geneFlow/figures/umap.png \
        --perts LZTR1 NDUFB6 MED13 KAT2A
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
import matplotlib.patches as mpatches
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
    """Run RK4 ODE and save intermediate states.

    Returns: (n_save + 1, B, n_genes) — start state + n_save intermediates,
    with baseline_lfc added back at every saved point.
    """
    device = x0.device
    B = x0.shape[0]
    if esm_emb.dim() == 1:
        esm_emb = esm_emb.unsqueeze(0).expand(B, -1)

    save_every = max(1, n_steps // n_save)
    x = x0.clone().float()
    dt = 1.0 / n_steps

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
        if (i + 1) % save_every == 0 or i == n_steps - 1:
            saved.append(x.clone() + model.baseline_lfc)

    return torch.stack(saved, dim=0)  # (T, B, n_genes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--out",    default="umap.png")
    parser.add_argument("--perts",  nargs="+",
                        default=["LZTR1", "NDUFB6", "MED13", "KAT2A"])
    parser.add_argument("--n_ctrl", type=int, default=300,
                        help="Control cells included in the background")
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

    perts = [p for p in args.perts if np.any(val_genes == p)]
    if not perts:
        raise ValueError(f"None of {args.perts} found in val perturbations")
    print(f"Plotting perturbations: {perts}")

    # ------------------------------------------------------------------
    # Collect cell populations
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    ctrl_idx = np.where(val_ctrl_mask)[0]
    ctrl_idx = rng.choice(ctrl_idx, min(args.n_ctrl, len(ctrl_idx)), replace=False)
    ctrl_cells = val_log1p[ctrl_idx]                        # (n_ctrl, G)

    # n_traj of the controls are the seeds for trajectories
    seed_idx = rng.choice(len(ctrl_cells), args.n_traj, replace=False)
    seed_cells = ctrl_cells[seed_idx]                       # (n_traj, G)
    seed_t = torch.from_numpy(seed_cells).float().to(device)

    blocks = {"control": ctrl_cells}
    traj_by_pert = {}
    true_by_pert = {}

    for pert in perts:
        mask = (~val_ctrl_mask) & (val_genes == pert)
        true_by_pert[pert] = val_log1p[mask]                # (n_true, G)
        esm_emb = gene_emb_map.get(pert, torch.zeros(cfg["esm_dim"])).to(device)
        traj = integrate_trajectory(
            model, seed_t, esm_emb,
            n_steps=args.ode_steps, n_save=args.n_save,
        ).cpu().numpy()                                     # (T, n_traj, G)
        traj_by_pert[pert] = traj
        blocks[f"{pert}__true"] = true_by_pert[pert]
        blocks[f"{pert}__traj"] = traj.reshape(-1, traj.shape[-1])

    # ------------------------------------------------------------------
    # Stack everything, fit PCA + UMAP once
    # ------------------------------------------------------------------
    names = list(blocks.keys())
    sizes = [blocks[n].shape[0] for n in names]
    X = np.concatenate([blocks[n] for n in names], axis=0)
    print(f"Total points for UMAP: {X.shape[0]} ({dict(zip(names, sizes))})")

    print("PCA → 50 components...")
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X)

    print("UMAP → 2D...")
    X_umap = UMAP(
        n_components=2, random_state=42,
        n_neighbors=30, min_dist=0.3,
    ).fit_transform(X_pca)

    # Split back into named slices
    offsets = np.cumsum([0] + sizes)
    slices = {n: X_umap[offsets[i]:offsets[i + 1]] for i, n in enumerate(names)}

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))

    pert_colors = plt.cm.tab10(np.linspace(0, 1, len(perts)))
    pert_color_map = dict(zip(perts, pert_colors))

    # Background: control cells
    ctrl_xy = slices["control"]
    ax.scatter(
        ctrl_xy[:, 0], ctrl_xy[:, 1],
        c="lightgrey", s=12, alpha=0.5, marker="o",
        linewidths=0, zorder=1, label=None,
    )

    for pert in perts:
        color = pert_color_map[pert]

        # True perturbed cells (triangles)
        true_xy = slices[f"{pert}__true"]
        ax.scatter(
            true_xy[:, 0], true_xy[:, 1],
            c=[color], s=35, alpha=0.7, marker="^",
            edgecolors="black", linewidths=0.4, zorder=2,
        )

        # Trajectories (lines + endpoint diamonds)
        traj_xy = slices[f"{pert}__traj"].reshape(
            args.n_save + 1, args.n_traj, 2
        )  # (T, n_traj, 2)

        for i in range(args.n_traj):
            ax.plot(
                traj_xy[:, i, 0], traj_xy[:, i, 1],
                c=color, alpha=0.6, linewidth=0.8, zorder=3,
            )
        # Start diamonds (brown, all perts share)
        ax.scatter(
            traj_xy[0, :, 0], traj_xy[0, :, 1],
            c="brown", s=70, marker="D", edgecolors="black", linewidths=0.5,
            zorder=4,
        )
        # End diamonds (per-pert color)
        ax.scatter(
            traj_xy[-1, :, 0], traj_xy[-1, :, 1],
            c=[color], s=70, marker="D", edgecolors="black", linewidths=0.5,
            zorder=4,
        )

    # Legends
    pert_patches = [mpatches.Patch(color=pert_color_map[p], label=p) for p in perts]
    type_handles = [
        plt.scatter([], [], c="lightgrey", s=30, marker="o", label="Control"),
        plt.scatter([], [], c="grey", s=40, marker="^",
                    edgecolors="black", linewidths=0.4, label="True perturbed"),
        plt.scatter([], [], c="brown", s=60, marker="D",
                    edgecolors="black", linewidths=0.5, label="Predicted start (x₀)"),
        plt.scatter([], [], c="grey", s=60, marker="D",
                    edgecolors="black", linewidths=0.5, label="Predicted end (x₁)"),
        plt.Line2D([], [], c="grey", linewidth=1.2, label="Predicted trajectory"),
    ]
    leg1 = ax.legend(handles=pert_patches, title="Perturbation",
                     loc="upper left", fontsize=9, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=type_handles, title="Cell type",
              loc="upper right", fontsize=9, frameon=True)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "Predicted cell-state trajectories under CRISPRi perturbation\n"
        f"(ODE integrated in 4721-dim log1p space, projected via PCA → UMAP)"
    )
    ax.spines[["top", "right"]].set_visible(False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
