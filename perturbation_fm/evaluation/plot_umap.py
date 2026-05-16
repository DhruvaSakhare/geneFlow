"""UMAP visualization of control, true perturbed, and predicted cell states.

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
from typing import List

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--out",    default="umap.png")
    parser.add_argument("--perts",  nargs="+", default=["LZTR1", "NDUFB6", "MED13", "KAT2A"],
                        help="Which perturbations to highlight")
    parser.add_argument("--n_ctrl", type=int, default=200,
                        help="Number of control cells to include")
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

    # Compute and set baseline LFC
    train_log1p = data_dict["train_log1p"]
    train_genes_arr = np.asarray(data_dict["train_genes"])
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    ctrl_mean = data_dict["ctrl_mean_log1p"]
    train_perts_unique = np.unique(train_genes_arr[~train_ctrl_mask])
    _lfcs = []
    for _p in train_perts_unique:
        _m = (~train_ctrl_mask) & (train_genes_arr == _p)
        _lfcs.append(train_log1p[_m].mean(axis=0) - ctrl_mean)
    baseline_lfc_vec = np.mean(_lfcs, axis=0)
    model.set_baseline_lfc(torch.from_numpy(baseline_lfc_vec).to(device))

    val_log1p   = data_dict["val_log1p"]
    val_genes   = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]

    # Filter to requested perturbations that exist in val
    perts = [p for p in args.perts if np.any(val_genes == p)]
    if not perts:
        raise ValueError(f"None of {args.perts} found in val perturbations")
    print(f"Plotting perturbations: {perts}")

    # Collect control cells (shared background)
    ctrl_idx = np.where(val_ctrl_mask)[0]
    rng = np.random.default_rng(42)
    ctrl_idx = rng.choice(ctrl_idx, min(args.n_ctrl, len(ctrl_idx)), replace=False)
    ctrl_cells = val_log1p[ctrl_idx]  # (n_ctrl, n_genes)

    # Collect true and predicted cells per perturbation
    all_cells = [ctrl_cells]
    labels = ["control"] * len(ctrl_cells)
    cell_types = ["control"] * len(ctrl_cells)  # control / true / predicted

    for pert in perts:
        mask = (~val_ctrl_mask) & (val_genes == pert)
        true_cells = val_log1p[mask]  # (n_pert, n_genes)

        # Predict: use control cells as x0
        x0 = torch.from_numpy(ctrl_cells).float().to(device)
        esm_emb = gene_emb_map.get(pert, torch.zeros(cfg["esm_dim"]))
        with torch.no_grad():
            out = model.predict(x0, esm_emb.to(device), steps=args.ode_steps)
        pred_cells = out["x1_log1p"].cpu().numpy()

        all_cells.append(true_cells)
        labels.extend([pert] * len(true_cells))
        cell_types.extend(["true"] * len(true_cells))

        all_cells.append(pred_cells)
        labels.extend([pert] * len(pred_cells))
        cell_types.extend(["predicted"] * len(pred_cells))

    X = np.concatenate(all_cells, axis=0)
    labels = np.array(labels)
    cell_types = np.array(cell_types)
    print(f"Total cells for UMAP: {X.shape[0]}")

    # PCA → UMAP
    print("Running PCA...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    print("Running UMAP...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    X_umap = reducer.fit_transform(X_pca)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    pert_colors = plt.cm.tab10(np.linspace(0, 1, len(perts)))
    pert_color_map = dict(zip(perts, pert_colors))

    markers = {"control": "o", "true": "^", "predicted": "x"}
    marker_sizes = {"control": 15, "true": 40, "predicted": 40}

    # Draw control cells first (grey background)
    ctrl_mask_plot = cell_types == "control"
    ax.scatter(
        X_umap[ctrl_mask_plot, 0], X_umap[ctrl_mask_plot, 1],
        c="lightgrey", s=marker_sizes["control"], alpha=0.5,
        marker=markers["control"], linewidths=0, zorder=1,
    )

    # Draw true and predicted per perturbation
    for pert, color in pert_color_map.items():
        for ctype, zorder in [("true", 3), ("predicted", 4)]:
            mask_plot = (labels == pert) & (cell_types == ctype)
            if not mask_plot.any():
                continue
            ax.scatter(
                X_umap[mask_plot, 0], X_umap[mask_plot, 1],
                c=[color], s=marker_sizes[ctype], alpha=0.8,
                marker=markers[ctype], linewidths=0.5,
                edgecolors="black" if ctype == "true" else "none",
                zorder=zorder,
            )

    # Legend — perturbations
    pert_patches = [
        mpatches.Patch(color=pert_color_map[p], label=p) for p in perts
    ]
    # Legend — cell types
    type_handles = [
        plt.scatter([], [], c="grey", s=30, marker="o", label="Control"),
        plt.scatter([], [], c="grey", s=30, marker="^", label="True perturbed",
                    edgecolors="black", linewidths=0.5),
        plt.scatter([], [], c="grey", s=30, marker="x", label="Predicted"),
    ]
    legend1 = ax.legend(handles=pert_patches, title="Perturbation",
                        loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=type_handles, title="Cell type",
              loc="upper right", fontsize=8)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Cell state transitions: control → perturbed\n"
                 "(△ true, × predicted, ○ control)")
    ax.spines[["top", "right"]].set_visible(False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
