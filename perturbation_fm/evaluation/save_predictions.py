"""Run the model's ODE for every val perturbation and save predicted cell
states at t = 0.0, 0.25, 0.50, 0.75, 1.0 as separate AnnData files.

Output files (written to --out_dir):
    predictions_t_0.00.h5ad
    predictions_t_0.25.h5ad
    predictions_t_0.50.h5ad
    predictions_t_0.75.h5ad
    predictions_t_1.00.h5ad

Each file contains:
    X            — (n_perts * n_seeds, n_genes) log1p expression
    obs['condition'] — perturbation name for each row ('control' for raw seeds)
    var          — gene-name index

Usage:
    python -m perturbation_fm.evaluation.save_predictions \\
        --config perturbation_fm/configs/default.yaml \\
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt \\
        --out_dir /work3/s225191/geneFlow/predictions/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml

from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.evaluation.plot_umap import integrate_at_times
from perturbation_fm.model.full_model import PerturbationFlowModel


TIME_POINTS = [0.00, 0.25, 0.50, 0.75, 1.00]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_seeds", type=int, default=200,
                        help="Control cells used as ODE seeds per perturbation")
    parser.add_argument("--ode_steps", type=int, default=50)
    parser.add_argument("--include_control", action="store_true",
                        help="Also write rows with condition='control' (raw seed cells)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
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

    # Restore baseline_lfc from training data (matches plot_umap.py)
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
    gene_names    = list(data_dict["gene_names"])

    perts = sorted(np.unique(val_genes[~val_ctrl_mask]).tolist())
    print(f"Generating predictions for {len(perts)} perturbations: {perts}")

    rng = np.random.default_rng(42)
    ctrl_idx = np.where(val_ctrl_mask)[0]
    ctrl_idx = rng.choice(ctrl_idx, min(args.n_seeds, len(ctrl_idx)), replace=False)
    ctrl_cells = val_log1p[ctrl_idx]
    seed_t = torch.from_numpy(ctrl_cells).float().to(device)
    print(f"ODE seed cells per perturbation: {len(ctrl_cells)}")

    # Integrate trajectories for every perturbation, capture all timepoints
    pred_by_pert_time: dict[str, dict[float, np.ndarray]] = {}
    for pert in perts:
        esm_emb = gene_emb_map.get(pert, torch.zeros(cfg["esm_dim"])).to(device)
        states = integrate_at_times(
            model, seed_t, esm_emb,
            times=TIME_POINTS, n_steps=args.ode_steps,
        )
        pred_by_pert_time[pert] = {t: states[t].cpu().numpy() for t in states}
        print(f"  {pert} done")

    # ------------------------------------------------------------------
    # Write one h5ad per timepoint
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var_df = pd.DataFrame(index=pd.Index(gene_names, name="gene"))

    for t in TIME_POINTS:
        blocks_X = []
        blocks_cond = []

        for pert in perts:
            X_p = pred_by_pert_time[pert][t]
            blocks_X.append(X_p)
            blocks_cond.extend([pert] * X_p.shape[0])

        if args.include_control:
            blocks_X.append(ctrl_cells.astype(np.float32))
            blocks_cond.extend(["control"] * ctrl_cells.shape[0])

        X = np.concatenate(blocks_X, axis=0).astype(np.float32)
        obs = pd.DataFrame({"condition": pd.Categorical(blocks_cond)})
        obs.index = [f"cell_{i:06d}" for i in range(X.shape[0])]

        adata = ad.AnnData(X=X, obs=obs, var=var_df)
        out_path = out_dir / f"predictions_t_{t:.2f}.h5ad"
        adata.write_h5ad(out_path)
        print(f"Wrote {out_path}  shape={adata.shape}")


if __name__ == "__main__":
    main()
