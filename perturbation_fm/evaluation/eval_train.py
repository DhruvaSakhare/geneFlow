"""Evaluate trained model on TRAINING perturbations.

Diagnostic: if PDS is high on train but low on val, the model learned
training-specific signal but doesn't generalize. If PDS is low on both,
the model has collapsed and isn't learning anything per-perturbation.

Usage:
    python -m perturbation_fm.evaluation.eval_train \
        --config perturbation_fm/configs/default.yaml \
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.evaluation.metrics import (
    evaluate_perturbation,
    perturbation_discrimination_score,
)
from perturbation_fm.model.full_model import PerturbationFlowModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt",   required=True, help="Path to best_model.pt")
    parser.add_argument("--n_perts", type=int, default=30,
                        help="Sample N training perturbations (default 30)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data_dict = preprocess(cfg["train_path"], cfg["val_path"])
    n_genes = data_dict["n_genes"]

    # Load embeddings
    gene_emb_map = torch.load(cfg["emb_path"], map_location="cpu")

    # Build model & load checkpoint
    model = PerturbationFlowModel(
        n_genes=n_genes,
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
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_lfc={ckpt['val_lfc']:.4f})")

    # Set up training data as if it were val
    train_log1p = data_dict["train_log1p"]
    train_counts = data_dict["train_counts"]
    train_genes_arr = np.asarray(data_dict["train_genes"])
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    ctrl_mean = data_dict["ctrl_mean_log1p"]
    control_log1p = train_log1p[train_ctrl_mask]
    gene_names_list = data_dict["gene_names"]

    train_perts = np.unique(train_genes_arr[~train_ctrl_mask])
    rng = np.random.default_rng(42)
    if len(train_perts) > args.n_perts:
        train_perts = rng.choice(train_perts, args.n_perts, replace=False)
    print(f"\nEvaluating on {len(train_perts)} training perturbations\n")

    per_pert: Dict[str, Dict] = {}
    for gene in train_perts:
        mask = (~train_ctrl_mask) & (train_genes_arr == gene)
        true_pert_log1p = train_log1p[mask]
        true_pert_counts = train_counts[mask]
        if len(true_pert_log1p) == 0:
            continue

        esm_emb = gene_emb_map.get(gene, torch.zeros(list(gene_emb_map.values())[0].shape[0]))
        metrics = evaluate_perturbation(
            model=model,
            control_log1p=control_log1p,
            true_pert_log1p=true_pert_log1p,
            true_pert_counts=true_pert_counts,
            ctrl_mean_log1p=ctrl_mean,
            esm_emb=esm_emb,
            gene_name=gene,
            gene_names_list=gene_names_list,
            device=str(device),
        )
        per_pert[gene] = metrics

    # Aggregate
    numeric_keys = ["pearson_r", "weighted_cos", "jaccard_top50", "e_distance", "variance_corr", "sliced_w2"]
    means = {k: float(np.mean([p[k] for p in per_pert.values()])) for k in numeric_keys}

    pred_lfcs = {g: m["_pred_lfc"] for g, m in per_pert.items()}
    true_lfcs = {g: m["_true_lfc"] for g, m in per_pert.items()}
    train_pds = perturbation_discrimination_score(pred_lfcs, true_lfcs)

    print("=" * 70)
    print(f"TRAINING perturbation evaluation ({len(per_pert)} perturbations)")
    print("=" * 70)
    print(f"Mean Pearson r:   {means['pearson_r']:.4f}")
    print(f"Mean Wtd-cos:     {means['weighted_cos']:.4f}")
    print(f"Mean Jaccard@50:  {means['jaccard_top50']:.4f}")
    print(f"Mean E-dist:      {means['e_distance']:.4f}")
    print(f"Mean SW2:         {means['sliced_w2']:.4f}")
    print(f"Mean Var-corr:    {means['variance_corr']:.4f}")
    print(f"TRAIN PDS:        {train_pds:.4f}  [1.0 = perfect, 0.5 = random]")


if __name__ == "__main__":
    main()
