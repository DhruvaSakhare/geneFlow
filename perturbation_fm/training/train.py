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
from perturbation_fm.evaluation.metrics import evaluate_all_perturbations, pearson_lfc
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
    rs = []
    for pert in val_perts:
        mask = (~val_ctrl_mask) & (val_genes == pert)
        true_lfc = val_log1p[mask].mean(axis=0) - ctrl_mean
        rs.append(pearson_lfc(mean_train_lfc, true_lfc))

    mean_r = float(np.mean(rs))
    print(f"[Baseline] Mean Pearson r (mean-train-LFC vs val): {mean_r:.4f}")
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
    ).to(device)

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
    best_val_loss = float("inf")
    patience_counter = 0
    loss_history: Dict[str, list] = {"train_fm": [], "train_nb": [], "val_fm": [], "val_nb": []}

    print("\n=== Training ===")
    for epoch in range(1, cfg["n_epochs"] + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        train_fm_losses, train_nb_losses = [], []

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
            train_nb_losses.append(result["loss_nb"])

        scheduler.step()

        mean_fm = float(np.mean(train_fm_losses))
        mean_nb = float(np.mean(train_nb_losses))

        # ---- Validate ----
        model.eval()
        val_fm_losses, val_nb_losses = [], []
        with torch.no_grad():
            for x0, x1_log1p, x1_counts, esm_emb in val_loader:
                x0 = x0.to(device)
                x1_log1p = x1_log1p.to(device)
                x1_counts = x1_counts.to(device)
                esm_emb = esm_emb.to(device)
                result = model.compute_loss(x0, x1_log1p, x1_counts, esm_emb)
                val_fm_losses.append(result["loss_fm"])
                val_nb_losses.append(result["loss_nb"])

        val_fm   = float(np.mean(val_fm_losses))
        val_nb   = float(np.mean(val_nb_losses))
        val_loss = val_fm + cfg["lambda_nb"] * val_nb
        elapsed  = time.time() - t0

        print(
            f"Epoch {epoch:4d}/{cfg['n_epochs']} | "
            f"train FM: {mean_fm:.4f}  NB: {mean_nb:.4f} | "
            f"val FM: {val_fm:.4f}  NB: {val_nb:.4f} | {elapsed:.1f}s"
        )

        loss_history["train_fm"].append(mean_fm)
        loss_history["train_nb"].append(mean_nb)
        loss_history["val_fm"].append(val_fm)
        loss_history["val_nb"].append(val_nb)

        # ---- Full perturbation eval every 5 epochs ----
        if epoch % 5 == 0:
            print(f"\n--- Perturbation eval (epoch {epoch}) ---")
            agg = evaluate_all_perturbations(
                model, data_dict, gene_emb_map, device=str(device)
            )
            m = agg["mean"]
            print(
                f"  mean Pearson r: {m['pearson_r']:.4f}  "
                f"mean E-dist: {m['e_distance']:.4f}\n"
            )
            model.train()

        # ---- Checkpointing — track FM loss only for early stopping ----
        # NB loss converges in epoch 1-2 and dominates the combined metric,
        # which would trigger early stopping before the flow network learns.
        if val_fm < best_val_loss:
            best_val_loss = val_fm
            patience_counter = 0
            ckpt_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_fm": val_fm,
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model  (val_fm={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch} (patience={cfg['patience']}).")
                break

    # Save loss history.
    with open(save_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


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
