"""Sanity check: skip the ODE entirely and only use baseline_lfc as prediction.

If this gives Pearson r ~0.24 (matching baseline), then the baseline_lfc
is being added correctly — the model's learned residual is what's making
things worse. If it gives something different, there's a bug.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import yaml

from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.evaluation.metrics import (
    pearson_lfc,
    weighted_cosine_lfc,
    perturbation_discrimination_score,
    diff_expression_score,
    mae_pseudobulk,
    vcc_perturbation_discrimination_score,
)
from perturbation_fm.model.full_model import PerturbationFlowModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--val_path", default=None,
                        help="Override val_path from config (e.g. point at test.h5ad)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_path = args.val_path or cfg["val_path"]
    print(f"Evaluating baseline against: {eval_path}")
    data_dict = preprocess(cfg["train_path"], eval_path)

    # Build model and load checkpoint just to get baseline_lfc
    model = PerturbationFlowModel(
        n_genes=data_dict["n_genes"], esm_dim=cfg["esm_dim"],
        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], lambda_nb=cfg["lambda_nb"],
        lambda_lfc=cfg["lambda_lfc"],
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    baseline_lfc = model.baseline_lfc.cpu().numpy()

    # Evaluate "predicting just baseline_lfc" on val perturbations
    val_log1p = data_dict["val_log1p"]
    val_genes = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]
    ctrl_mean = data_dict["ctrl_mean_log1p"]
    gene_names_list = list(data_dict["gene_names"])
    val_perts = np.unique(val_genes[~val_ctrl_mask])

    train_log1p = data_dict["train_log1p"]
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    train_control_log1p = train_log1p[train_ctrl_mask]

    # Sample control cells once — baseline "prediction" = these cells + baseline_lfc.
    # Same prediction is reused for every perturbation (that IS the baseline).
    rng = np.random.default_rng(42)
    n_sample = 1000
    ctrl_idx = rng.choice(
        len(train_control_log1p), n_sample,
        replace=len(train_control_log1p) < n_sample,
    )
    ctrl_sample = train_control_log1p[ctrl_idx]
    pred_perturbed_cells = ctrl_sample + baseline_lfc  # (n_sample, n_genes)

    rs, wcos, des_list, mae_list = [], [], [], []
    pred_lfcs, true_lfcs = {}, {}

    print("=" * 70)
    print("BASELINE-ONLY PREDICTION (no model residual — just baseline_lfc)")
    print("=" * 70)
    header = f"{'Perturbation':<20} {'Pearson r':>10} {'Wtd-cos':>10} {'DES':>8} {'MAE':>8}"
    print(header)
    print("-" * len(header))

    for pert in val_perts:
        mask = (~val_ctrl_mask) & (val_genes == pert)
        true_pert_cells = val_log1p[mask]
        if len(true_pert_cells) == 0:
            continue
        true_lfc = true_pert_cells.mean(axis=0) - ctrl_mean
        pred_lfc = baseline_lfc

        r = pearson_lfc(pred_lfc, true_lfc)
        wc = weighted_cosine_lfc(pred_lfc, true_lfc)
        des = diff_expression_score(
            pred_perturbed=pred_perturbed_cells,
            pred_control=ctrl_sample,
            true_perturbed=true_pert_cells,
            true_control=train_control_log1p,
            pred_lfc=pred_lfc,
        )
        mae = mae_pseudobulk(pred_perturbed_cells, true_pert_cells)

        rs.append(r); wcos.append(wc); des_list.append(des); mae_list.append(mae)
        pred_lfcs[pert] = pred_lfc
        true_lfcs[pert] = true_lfc

        print(f"{pert:<20} {r:>10.4f} {wc:>10.4f} {des:>8.4f} {mae:>8.4f}")

    print("-" * len(header))
    print(
        f"{'MEAN':<20} {float(np.mean(rs)):>10.4f} {float(np.mean(wcos)):>10.4f} "
        f"{float(np.nanmean(des_list)):>8.4f} {float(np.mean(mae_list)):>8.4f}"
    )
    print()
    print(f"PDS (ours, L2):  {perturbation_discrimination_score(pred_lfcs, true_lfcs):.4f}")
    print(f"PDS (VCC, L1):   {vcc_perturbation_discrimination_score(pred_lfcs, true_lfcs, gene_names_list):.4f}")
    print()
    print("These numbers scale the model's VCC score. The cell-mean baseline")
    print("predicts the SAME LFC for every perturbation, so PDS-VCC ≈ 0.5")
    print("(random) by construction; DES and MAE are the real bars to clear.")


if __name__ == "__main__":
    main()
