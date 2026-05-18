"""Run the model's ODE for every val/test perturbation and save predicted cell
states at t = 0.0, 0.25, 0.50, 0.75, 1.0 as separate AnnData files with
pre-computed UMAP coordinates in a shared embedding space.

All five timepoint files share the same PCA → UMAP fit, so cell positions
are directly comparable across panels.  Each file contains:
    X                  — (n_cells, n_genes) log1p expression
    obs['condition']   — perturbation name or 'control'
    obs['cell_type']   — 'predicted', 'control', or 'true'
    obsm['X_umap']     — 2-D UMAP coordinates (shared across files)

Usage:
    python -m perturbation_fm.evaluation.save_predictions \\
        --config perturbation_fm/configs/default.yaml \\
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt \\
        --out_dir /work3/s225191/geneFlow/predictions/ \\
        --data_path /work3/s225191/geneFlow/data/test.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from umap import UMAP

from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.evaluation.plot_umap import integrate_at_times
from perturbation_fm.model.full_model import PerturbationFlowModel


TIME_POINTS = [0.00, 0.25, 0.50, 0.75, 1.00]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",      required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--data_path", default=None,
                        help="Path to val or test h5ad. Defaults to val_path in config.")
    parser.add_argument("--n_ctrl_bg", type=int, default=None,
                        help="Max background control cells for UMAP (default: all)")
    parser.add_argument("--ode_steps", type=int, default=50)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    data_path = args.data_path or cfg["val_path"]
    data_dict = preprocess(cfg["train_path"], data_path)
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

    val_log1p     = data_dict["val_log1p"]
    val_genes     = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]
    gene_names    = list(data_dict["gene_names"])
    perts         = sorted(np.unique(val_genes[~val_ctrl_mask]).tolist())
    print(f"Perturbations ({len(perts)}): {perts}")

    rng = np.random.default_rng(42)
    all_ctrl_idx = np.where(val_ctrl_mask)[0]

    # Background control cloud — all available control cells (or capped)
    if args.n_ctrl_bg and args.n_ctrl_bg < len(all_ctrl_idx):
        bg_idx  = rng.choice(all_ctrl_idx, args.n_ctrl_bg, replace=False)
        ctrl_bg = val_log1p[bg_idx]
    else:
        ctrl_bg = val_log1p[all_ctrl_idx]
    print(f"Background control cells: {len(ctrl_bg)}")

    # ------------------------------------------------------------------
    # Integrate ODE for every perturbation; sample true perturbed cells
    # Use the same number of ODE seeds as true cells for that perturbation
    # ------------------------------------------------------------------
    pred: dict[str, dict[float, np.ndarray]] = {}
    true: dict[str, np.ndarray] = {}

    for pert in perts:
        pert_idx = np.where((~val_ctrl_mask) & (val_genes == pert))[0]
        true[pert] = val_log1p[pert_idx]   # all true cells, no subsampling
        n_cells = len(pert_idx)

        # Sample n_cells control cells as ODE seeds (with replacement if needed)
        seed_idx  = rng.choice(all_ctrl_idx, n_cells, replace=len(all_ctrl_idx) < n_cells)
        ctrl_cells = val_log1p[seed_idx]
        seed_t    = torch.from_numpy(ctrl_cells).float().to(device)

        esm_emb = gene_emb_map.get(pert, torch.zeros(cfg["esm_dim"])).to(device)
        states  = integrate_at_times(model, seed_t, esm_emb,
                                     times=TIME_POINTS, n_steps=args.ode_steps)
        pred[pert] = {t: states[t].cpu().numpy() for t in states}
        print(f"  {pert}: {n_cells} cells")

    # ------------------------------------------------------------------
    # Fit ONE shared PCA → UMAP on everything
    # ------------------------------------------------------------------
    print("\nFitting shared PCA → UMAP...")

    all_arrays = [ctrl_bg]
    for pert in perts:
        for t in TIME_POINTS:
            all_arrays.append(pred[pert][t])
        all_arrays.append(true[pert])

    X_all = np.concatenate(all_arrays, axis=0).astype(np.float32)
    sizes = [a.shape[0] for a in all_arrays]
    print(f"  Total cells for UMAP fit: {X_all.shape[0]}")

    pca     = PCA(n_components=50, random_state=42)
    X_pca   = pca.fit_transform(X_all)
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    X_umap  = reducer.fit_transform(X_pca)

    # Carve back into per-source arrays
    offsets = np.cumsum([0] + sizes)
    ctrl_bg_umap = X_umap[offsets[0]:offsets[1]]

    idx = 1
    pred_umap: dict[str, dict[float, np.ndarray]] = {p: {} for p in perts}
    true_umap: dict[str, np.ndarray] = {}
    for pert in perts:
        for t in TIME_POINTS:
            pred_umap[pert][t] = X_umap[offsets[idx]:offsets[idx + 1]]
            idx += 1
        true_umap[pert] = X_umap[offsets[idx]:offsets[idx + 1]]
        idx += 1

    # ------------------------------------------------------------------
    # Write one h5ad per timepoint
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var_df  = pd.DataFrame(index=pd.Index(gene_names, name="gene"))

    for t in TIME_POINTS:
        X_blocks, cond_list, type_list, umap_blocks = [], [], [], []

        # Background control cells
        X_blocks.append(ctrl_bg.astype(np.float32))
        cond_list.extend(["control"] * len(ctrl_bg))
        type_list.extend(["control"] * len(ctrl_bg))
        umap_blocks.append(ctrl_bg_umap)

        # Predicted cells at this timepoint
        for pert in perts:
            X_p = pred[pert][t]
            X_blocks.append(X_p.astype(np.float32))
            cond_list.extend([pert] * X_p.shape[0])
            type_list.extend(["predicted"] * X_p.shape[0])
            umap_blocks.append(pred_umap[pert][t])

        # True perturbed cells (only in t=1.0 file; grayed in others)
        for pert in perts:
            X_blocks.append(true[pert].astype(np.float32))
            cond_list.extend([f"{pert}_true"] * true[pert].shape[0])
            type_list.extend(["true"] * true[pert].shape[0])
            umap_blocks.append(true_umap[pert])

        X      = np.concatenate(X_blocks, axis=0)
        umap_coords = np.concatenate(umap_blocks, axis=0)
        obs    = pd.DataFrame({
            "condition": pd.Categorical(cond_list),
            "cell_type": pd.Categorical(type_list),
        })
        obs.index = [f"cell_{i:07d}" for i in range(X.shape[0])]

        adata  = ad.AnnData(X=X, obs=obs, var=var_df)
        adata.obsm["X_umap"] = umap_coords

        out_path = out_dir / f"predictions_t_{t:.2f}.h5ad"
        adata.write_h5ad(out_path)
        print(f"Wrote {out_path}  shape={adata.shape}")

    print("\nDone. All files share the same UMAP embedding.")


if __name__ == "__main__":
    main()
