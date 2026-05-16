"""Training script for PerturbationFlowModel.

Run from the repository root:
    python -m perturbation_fm.training.train [--config path/to/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from perturbation_fm.data.dataset import PerturbationDataset, get_dataloader
from perturbation_fm.data.preprocess import preprocess
from perturbation_fm.evaluation.metrics import evaluate_all_perturbations, pearson_lfc, weighted_cosine_lfc
from perturbation_fm.model.full_model import PerturbationFlowModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def baseline(data_dict: Dict) -> float:
    """Evaluate a constant-prediction baseline on validation perturbations.

    The prediction is the *mean LFC across all 150 training perturbations*.
    Prints the mean Pearson r against all 50 validation perturbations.

    Returns:
        Mean Pearson r (float).
    """
    train_log1p = data_dict["train_log1p"]
    train_genes = np.asarray(data_dict["train_genes"])
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    ctrl_mean = data_dict["ctrl_mean_log1p"]

    # Compute LFC for each training perturbation.
    train_perts = np.unique(train_genes[~train_ctrl_mask])
    train_lfcs = []
    for pert in train_perts:
        mask = (~train_ctrl_mask) & (train_genes == pert)
        lfc = train_log1p[mask].mean(axis=0) - ctrl_mean
        train_lfcs.append(lfc)
    mean_train_lfc = np.mean(train_lfcs, axis=0)  # (n_genes,)

    # Evaluate against each validation perturbation.
    val_log1p = data_dict["val_log1p"]
    val_genes = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]

    val_perts = np.unique(val_genes[~val_ctrl_mask])
    rs, wcos = [], []
    for pert in val_perts:
        mask = (~val_ctrl_mask) & (val_genes == pert)
        true_lfc = val_log1p[mask].mean(axis=0) - ctrl_mean
        rs.append(pearson_lfc(mean_train_lfc, true_lfc))
        wcos.append(weighted_cosine_lfc(mean_train_lfc, true_lfc))

    mean_r    = float(np.mean(rs))
    mean_wcos = float(np.mean(wcos))
    print(f"[Baseline] Mean Pearson r: {mean_r:.4f}  Weighted cosine: {mean_wcos:.4f}")
    return mean_r


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Dict) -> None:
    _set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n=== Preprocessing ===")
    data_dict = preprocess(cfg["train_path"], cfg["val_path"])

    n_genes = data_dict["n_genes"]
    cfg["n_genes"] = n_genes  # fill in the placeholder

    print(f"\n=== Loading ESM-2 embeddings from {cfg['emb_path']} ===")
    emb_path = Path(cfg["emb_path"])
    if not emb_path.exists():
        raise FileNotFoundError(
            f"ESM-2 embeddings not found at {emb_path}. "
            "Run embeddings/esm2_embed.py first."
        )
    gene_emb_map: Dict[str, torch.Tensor] = torch.load(emb_path, map_location="cpu")

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_dataset = PerturbationDataset(
        log1p_data=data_dict["train_log1p"],
        counts_data=data_dict["train_counts"],
        gene_targets=data_dict["train_genes"],
        ctrl_mask=data_dict["train_ctrl_mask"],
        gene_emb_map=gene_emb_map,
    )
    val_dataset = PerturbationDataset(
        log1p_data=data_dict["val_log1p"],
        counts_data=data_dict["val_counts"],
        gene_targets=data_dict["val_genes"],
        ctrl_mask=data_dict["val_ctrl_mask"],
        gene_emb_map=gene_emb_map,
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
        balanced=True,
        n_perts_per_batch=cfg["n_perts_per_batch"],
        n_cells_per_pert=cfg["n_cells_per_pert"],
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print("\n=== Baseline ===")
    baseline(data_dict)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PerturbationFlowModel(
        n_genes=n_genes,
        esm_dim=cfg["esm_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        lambda_nb=cfg["lambda_nb"],
        lambda_lfc=cfg["lambda_lfc"],
    ).to(device)

    train_log1p_full = data_dict["train_log1p"]
    train_genes_arr = np.asarray(data_dict["train_genes"])
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    ctrl_mean_train = data_dict["ctrl_mean_log1p"]
    train_perts_unique = np.unique(train_genes_arr[~train_ctrl_mask])
    _train_lfcs = []
    for _pert in train_perts_unique:
        _mask = (~train_ctrl_mask) & (train_genes_arr == _pert)
        _train_lfcs.append(train_log1p_full[_mask].mean(axis=0) - ctrl_mean_train)
    baseline_lfc_vec = np.mean(_train_lfcs, axis=0)
    model.set_baseline_lfc(torch.from_numpy(baseline_lfc_vec).to(device))
    print(f"Baseline LFC computed: mean |LFC| = {np.abs(baseline_lfc_vec).mean():.4f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_pearson = -float("inf")
    loss_history: Dict[str, list] = {"train_fm": [], "train_lfc": [], "val_fm": [], "val_lfc": []}

    print("\n=== Training ===")
    for epoch in range(1, cfg["n_epochs"] + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        train_fm_losses, train_lfc_losses = [], []

        for x0, x1_log1p, x1_counts, esm_emb in train_loader:
            x0 = x0.to(device)
            x1_log1p = x1_log1p.to(device)
            x1_counts = x1_counts.to(device)
            esm_emb = esm_emb.to(device)

            optimizer.zero_grad()
            result = model.compute_loss(x0, x1_log1p, x1_counts, esm_emb)
            result["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            train_fm_losses.append(result["loss_fm"])
            train_lfc_losses.append(result["loss_lfc"])

        scheduler.step()

        mean_fm  = float(np.mean(train_fm_losses))
        mean_lfc = float(np.mean(train_lfc_losses))

        # ---- Validate ----
        model.eval()
        val_fm_losses, val_lfc_losses = [], []
        with torch.no_grad():
            for x0, x1_log1p, x1_counts, esm_emb in val_loader:
                x0 = x0.to(device)
                x1_log1p = x1_log1p.to(device)
                x1_counts = x1_counts.to(device)
                esm_emb = esm_emb.to(device)
                result = model.compute_loss(x0, x1_log1p, x1_counts, esm_emb)
                val_fm_losses.append(result["loss_fm"])
                val_lfc_losses.append(result["loss_lfc"])

        val_fm  = float(np.mean(val_fm_losses))
        val_lfc = float(np.mean(val_lfc_losses))
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:4d}/{cfg['n_epochs']} | "
            f"train FM: {mean_fm:.4f}  LFC: {mean_lfc:.4f} | "
            f"val FM: {val_fm:.4f}  LFC: {val_lfc:.4f} | {elapsed:.1f}s"
        )

        loss_history["train_fm"].append(mean_fm)
        loss_history["train_lfc"].append(mean_lfc)
        loss_history["val_fm"].append(val_fm)
        loss_history["val_lfc"].append(val_lfc)

        # ---- Full perturbation eval every 5 epochs + checkpoint on Pearson r ----
        if epoch % 5 == 0:
            print(f"\n--- Perturbation eval (epoch {epoch}) ---")
            agg = evaluate_all_perturbations(
                model, data_dict, gene_emb_map, device=str(device)
            )
            m = agg["mean"]
            val_pearson = m["pearson_r"]
            val_pds = m["pds"]
            # Combined score: Pearson r + (PDS - 0.5) puts both on similar scale
            val_score = val_pearson + (val_pds - 0.5)
            print(
                f"  mean Pearson r: {val_pearson:.4f}  "
                f"PDS: {val_pds:.4f}  score: {val_score:.4f}  "
                f"mean E-dist: {m['e_distance']:.4f}\n"
            )
            if val_score > best_val_pearson:
                best_val_pearson = val_score
                ckpt_path = save_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_pearson": val_pearson,
                        "val_pds": val_pds,
                        "val_score": val_score,
                        "cfg": cfg,
                    },
                    ckpt_path,
                )
                print(f"  ✓ Saved best model  (score={best_val_pearson:.4f})")
            model.train()

    # Save loss history.
    with open(save_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nTraining complete. Best val score (Pearson r + PDS - 0.5): {best_val_pearson:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PerturbationFlowModel.")
    parser.add_argument(
        "--config",
        default="perturbation_fm/configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
