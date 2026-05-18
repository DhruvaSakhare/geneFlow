"""Cell-mean baseline: predict the same mean-training-LFC for every perturbation.

No model weights are used — this is the pure cell-mean baseline that the VCC
composite uses as its reference point. Accepts val (default) or test h5ad.

Usage:
    # Validation set
    python -m perturbation_fm.evaluation.eval_baseline_only \\
        --config perturbation_fm/configs/default.yaml

    # Test set
    python -m perturbation_fm.evaluation.eval_baseline_only \\
        --config perturbation_fm/configs/default.yaml \\
        --eval_path /work3/s225191/geneFlow/data/test.h5ad
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.evaluation.metrics import (
    diff_expression_score,
    mae_pseudobulk,
    pearson_lfc,
    vcc_perturbation_discrimination_score,
    weighted_cosine_lfc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--eval_path", default=None,
                        help="Path to val or test h5ad. Defaults to val_path in config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_path = args.eval_path or cfg["val_path"]
    print(f"Evaluating cell-mean baseline against: {eval_path}")

    data_dict = preprocess(cfg["train_path"], eval_path)

    # Compute baseline_lfc = mean LFC across all training perturbations
    train_log1p      = data_dict["train_log1p"]
    train_genes_arr  = np.asarray(data_dict["train_genes"])
    train_ctrl_mask  = data_dict["train_ctrl_mask"]
    ctrl_mean        = data_dict["ctrl_mean_log1p"]

    train_perts = np.unique(train_genes_arr[~train_ctrl_mask])
    lfcs = [
        train_log1p[(~train_ctrl_mask) & (train_genes_arr == p)].mean(axis=0) - ctrl_mean
        for p in train_perts
    ]
    baseline_lfc = np.mean(lfcs, axis=0)   # (n_genes,)

    # Sample control cells from training set (baseline prediction = ctrl + baseline_lfc)
    rng = np.random.default_rng(42)
    train_ctrl_cells = train_log1p[train_ctrl_mask]
    n_sample = min(1000, len(train_ctrl_cells))
    ctrl_sample = train_ctrl_cells[rng.choice(len(train_ctrl_cells), n_sample, replace=False)]
    pred_cells = ctrl_sample + baseline_lfc   # same prediction for every perturbation

    # Evaluate on held-out perturbations
    val_log1p     = data_dict["val_log1p"]
    val_genes     = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]
    gene_names    = list(data_dict["gene_names"])
    eval_perts    = sorted(np.unique(val_genes[~val_ctrl_mask]).tolist())

    rs, wcos, des_list, mae_list = [], [], [], []
    pred_lfcs: dict[str, np.ndarray] = {}
    true_lfcs: dict[str, np.ndarray] = {}

    print("=" * 70)
    print("CELL-MEAN BASELINE (same LFC predicted for every perturbation)")
    print("=" * 70)
    hdr = f"{'Perturbation':<20} {'Pearson r':>10} {'Wtd-cos':>10} {'DES':>8} {'MAE':>8}"
    print(hdr)
    print("-" * len(hdr))

    for pert in eval_perts:
        mask = (~val_ctrl_mask) & (val_genes == pert)
        true_pert = val_log1p[mask]
        if len(true_pert) == 0:
            continue
        true_lfc = true_pert.mean(axis=0) - ctrl_mean

        r   = pearson_lfc(baseline_lfc, true_lfc)
        wc  = weighted_cosine_lfc(baseline_lfc, true_lfc)
        des = diff_expression_score(
            pred_perturbed=pred_cells,
            pred_control=ctrl_sample,
            true_perturbed=true_pert,
            true_control=train_ctrl_cells,
            pred_lfc=baseline_lfc,
        )
        mae = mae_pseudobulk(pred_cells, true_pert)

        rs.append(r); wcos.append(wc); des_list.append(des); mae_list.append(mae)
        pred_lfcs[pert] = baseline_lfc
        true_lfcs[pert] = true_lfc
        print(f"{pert:<20} {r:>10.4f} {wc:>10.4f} {des:>8.4f} {mae:>8.4f}")

    print("-" * len(hdr))
    mean_r   = float(np.mean(rs))
    mean_wc  = float(np.mean(wcos))
    mean_des = float(np.nanmean(des_list))
    mean_mae = float(np.mean(mae_list))
    pds      = vcc_perturbation_discrimination_score(pred_lfcs, true_lfcs, gene_names)

    print(f"{'MEAN':<20} {mean_r:>10.4f} {mean_wc:>10.4f} {mean_des:>8.4f} {mean_mae:>8.4f}")
    print()
    print(f"PDS (VCC):  {pds:.4f}")
    print()
    print("Note: PDS-VCC ≈ (N+1)/(2N) = 0.533 for N=15 by construction")
    print("      because all predictions are identical (same rank distribution).")


if __name__ == "__main__":
    main()
