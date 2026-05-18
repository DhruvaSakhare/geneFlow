"""Final evaluation on the held-out test set.

Loads a trained checkpoint and runs the full eval pipeline on test
perturbations the model has never seen. Reports all VCC metrics.

Usage:
    python -m perturbation_fm.evaluation.eval_test \\
        --config perturbation_fm/configs/default.yaml \\
        --ckpt /work3/s225191/geneFlow/checkpoints/best_model.pt \\
        --test_path /work3/s225191/geneFlow/data/test.h5ad
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.evaluation.metrics import evaluate_all_perturbations
from perturbation_fm.model.full_model import PerturbationFlowModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perturbation_fm/configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_path", required=True,
                        help="Path to the held-out test h5ad")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Preprocess with test.h5ad in the val slot — same code path, same gene
    # selection (driven by training perturbations), correctly held out.
    data_dict = preprocess(cfg["train_path"], args.test_path)
    gene_emb_map = torch.load(cfg["emb_path"], map_location="cpu")

    test_perts = sorted(np.unique(
        np.asarray(data_dict["val_genes"])[~data_dict["val_ctrl_mask"]]
    ).tolist())
    print(f"\nTest perturbations ({len(test_perts)}): {test_perts}")

    # Sanity: warn if any test gene overlaps training set
    train_perts = set(np.unique(
        np.asarray(data_dict["train_genes"])[~data_dict["train_ctrl_mask"]]
    ).tolist())
    overlap = sorted(set(test_perts) & train_perts)
    if overlap:
        print(f"  ⚠ {len(overlap)} test perts also appear in training: {overlap}")
    else:
        print("  ✓ All test perturbations are unseen by the model.")

    # Build & load model
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
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Restore baseline_lfc from training perturbations
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

    # Warn for test genes with missing ESM embeddings
    missing_emb = [g for g in test_perts if g not in gene_emb_map]
    if missing_emb:
        print(f"\n  ⚠ {len(missing_emb)} test genes lack ESM embeddings "
              f"(predictions will use zero vectors): {missing_emb}")
        print("    → Run add_val_embeddings.py with the test h5ad to fix this.")

    print("\n=== Running full test-set evaluation (with DES) ===\n")
    agg = evaluate_all_perturbations(
        model, data_dict, gene_emb_map, device=str(device),
        compute_des=True,
    )

    m = agg["mean"]
    print("\n" + "=" * 70)
    print("HELD-OUT TEST-SET RESULTS")
    print("=" * 70)
    print(f"Pearson r       : {m['pearson_r']:.4f}")
    print(f"Wtd-cos         : {m['weighted_cos']:.4f}")
    print(f"PDS (VCC)       : {m['pds_vcc']:.4f}")
    print(f"DES (VCC)       : {m['des']:.4f}")
    print(f"MAE (VCC)       : {m['mae']:.4f}")
    print(f"KD-ok rate      : {m['knockdown_ok_rate']:.4f}")


if __name__ == "__main__":
    main()
