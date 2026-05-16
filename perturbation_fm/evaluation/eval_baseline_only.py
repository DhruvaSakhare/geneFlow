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
)
from perturbation_fm.model.full_model import PerturbationFlowModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",   required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = preprocess(cfg["train_path"], cfg["val_path"])

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
    val_perts = np.unique(val_genes[~val_ctrl_mask])

    rs, wcos = [], []
    pred_lfcs, true_lfcs = {}, {}
    for pert in val_perts:
        mask = (~val_ctrl_mask) & (val_genes == pert)
        true_lfc = val_log1p[mask].mean(axis=0) - ctrl_mean
        pred_lfc = baseline_lfc  # the only prediction
        rs.append(pearson_lfc(pred_lfc, true_lfc))
        wcos.append(weighted_cosine_lfc(pred_lfc, true_lfc))
        pred_lfcs[pert] = pred_lfc
        true_lfcs[pert] = true_lfc

    print("=" * 60)
    print("BASELINE-ONLY PREDICTION (no model residual)")
    print("=" * 60)
    print(f"Mean Pearson r: {float(np.mean(rs)):.4f}")
    print(f"Mean Wtd-cos:   {float(np.mean(wcos)):.4f}")
    print(f"PDS:            {perturbation_discrimination_score(pred_lfcs, true_lfcs):.4f}")
    print()
    print("If this matches val baseline (0.2379 / 0.3726 / 0.5), then")
    print("baseline_lfc is correctly added — the model's residual is the")
    print("source of degraded performance.")


if __name__ == "__main__":
    main()
