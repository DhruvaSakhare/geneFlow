"""
Preprocessing pipeline for Virtual Cell Challenge 2025 (H1 hESC CRISPRi).

Raw integer counts -> log1p-normalised expression, LFC-based gene selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_lfc(
    pert_log1p_cells: np.ndarray,
    ctrl_mean_log1p: np.ndarray,
) -> np.ndarray:
    """Return per-gene log-fold-change for a perturbed cell population.

    Args:
        pert_log1p_cells: (M, n_genes) log1p-normalised expression of perturbed cells.
        ctrl_mean_log1p:  (n_genes,)   mean log1p-normalised expression of control cells.

    Returns:
        (n_genes,) LFC vector: mean(pert) - mean(ctrl).
    """
    return pert_log1p_cells.mean(axis=0) - ctrl_mean_log1p


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _to_dense(X) -> np.ndarray:
    """Convert sparse or dense matrix to float32 ndarray."""
    if issparse(X):
        return np.asarray(X.todense(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _normalize_log1p(counts: np.ndarray) -> np.ndarray:
    """Per-cell normalize to 10k counts then apply log1p.

    Args:
        counts: (n_cells, n_genes) raw integer counts.

    Returns:
        (n_cells, n_genes) float32 log1p-normalised expression.
    """
    total = counts.sum(axis=1, keepdims=True).astype(np.float32)
    # Guard against zero-count cells (shouldn't happen after QC, but be safe).
    total = np.where(total == 0, 1.0, total)
    normalized = counts.astype(np.float32) / total * 10_000.0
    return np.log1p(normalized)


def _select_genes_by_lfc(
    adata: AnnData,
    log1p_layer: np.ndarray,
    top_k: int = 200,
) -> List[str]:
    """Select the union of top-k DE genes across all training perturbations.

    For each perturbation:
      - Pseudo-bulk mean of perturbed cells in log1p space.
      - Pseudo-bulk mean of control cells in log1p space.
      - LFC = pert_mean - ctrl_mean
      - Keep top `top_k` genes by |LFC|.

    Returns:
        Sorted list of selected gene names.
    """
    ctrl_mask = np.asarray(adata.obs["target_gene"].values == "non-targeting", dtype=bool)
    ctrl_log1p = log1p_layer[ctrl_mask]           # (n_ctrl, n_genes)
    ctrl_mean = ctrl_log1p.mean(axis=0)            # (n_genes,)

    gene_names: np.ndarray = np.asarray(adata.var_names)
    selected: set[str] = set()

    perturbations = adata.obs.loc[~ctrl_mask, "target_gene"].unique()

    for pert in perturbations:
        pert_mask = (~ctrl_mask) & (adata.obs["target_gene"].values == pert)
        pert_log1p = log1p_layer[pert_mask]        # (n_pert, n_genes)

        if pert_log1p.shape[0] == 0:
            continue

        lfc = pert_log1p.mean(axis=0) - ctrl_mean  # (n_genes,)
        top_indices = np.argsort(np.abs(lfc))[-top_k:]
        selected.update(gene_names[top_indices].tolist())

    return sorted(selected)


def _subset_to_genes(adata: AnnData, log1p_layer: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log1p_subset, counts_subset) restricted to `gene_names`.

    Args:
        adata:       AnnData object (used for var_names and .X).
        log1p_layer: (n_cells, all_genes) log1p array aligned with adata.var_names.
        gene_names:  selected gene names (subset of adata.var_names).

    Returns:
        Tuple of (log1p, raw_counts) both shape (n_cells, len(gene_names)).
    """
    all_genes = list(adata.var_names)
    idx = [all_genes.index(g) for g in gene_names]
    log1p_sub = log1p_layer[:, idx]
    counts_sub = _to_dense(adata.X)[:, idx]
    return log1p_sub, counts_sub


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess(
    adata_path: str | Path,
    val_adata_path: str | Path,
    top_k_per_pert: int = 200,
) -> Dict:
    """Load, normalise, and select genes from H1 hESC CRISPRi data.

    Args:
        adata_path:       Path to training.h5ad.
        val_adata_path:   Path to validation.h5ad.
        top_k_per_pert:   Genes to keep per perturbation before taking the union.

    Returns:
        Dict with keys: train_log1p, train_counts, train_genes, train_ctrl_mask,
                        val_log1p, val_counts, val_genes, val_ctrl_mask,
                        gene_names, n_genes, ctrl_mean_log1p.
    """
    adata_path = Path(adata_path)
    val_adata_path = Path(val_adata_path)

    print(f"Loading training data from {adata_path} …")
    adata_train = sc.read_h5ad(adata_path)

    print(f"Loading validation data from {val_adata_path} …")
    adata_val = sc.read_h5ad(val_adata_path)

    # ------------------------------------------------------------------
    # 1. Raw counts
    # ------------------------------------------------------------------
    train_counts_all = _to_dense(adata_train.X)   # (n_train, all_genes)
    val_counts_all   = _to_dense(adata_val.X)     # (n_val,   all_genes)

    # ------------------------------------------------------------------
    # 2. Log1p normalisation (per-cell, before any averaging)
    # ------------------------------------------------------------------
    print("Normalising training cells …")
    train_log1p_all = _normalize_log1p(train_counts_all)

    print("Normalising validation cells …")
    val_log1p_all = _normalize_log1p(val_counts_all)

    # Store in a temporary layer so _select_genes_by_lfc can use it
    # without modifying the AnnData object on disk.

    # ------------------------------------------------------------------
    # 3. Gene selection on training data only
    # ------------------------------------------------------------------
    print("Selecting genes by LFC across training perturbations …")
    selected_genes = _select_genes_by_lfc(adata_train, train_log1p_all, top_k=top_k_per_pert)
    print(f"  → Selected {len(selected_genes)} genes (union of top-{top_k_per_pert} per perturbation).")

    # ------------------------------------------------------------------
    # 4. Subset to selected genes
    # ------------------------------------------------------------------
    # We need both datasets to share the same gene space; both were
    # measured on the same gene panel, so var_names should match.
    train_log1p, train_counts = _subset_to_genes(adata_train, train_log1p_all, selected_genes)
    val_log1p,   val_counts   = _subset_to_genes(adata_val,   val_log1p_all,   selected_genes)

    # ------------------------------------------------------------------
    # 5. Masks and metadata
    # ------------------------------------------------------------------
    train_ctrl_mask = np.asarray(adata_train.obs["target_gene"].values == "non-targeting", dtype=bool)
    val_ctrl_mask   = np.asarray(adata_val.obs["target_gene"].values   == "non-targeting", dtype=bool)

    train_genes = list(adata_train.obs["target_gene"].values)
    val_genes   = list(adata_val.obs["target_gene"].values)

    # Control mean in log1p space (on selected genes, training controls only)
    ctrl_mean_log1p = train_log1p[train_ctrl_mask].mean(axis=0)  # (n_genes,)

    print("Preprocessing complete.")
    print(f"  Train: {train_log1p.shape[0]} cells, {train_log1p.shape[1]} genes")
    print(f"  Val:   {val_log1p.shape[0]} cells,   {val_log1p.shape[1]} genes")
    print(f"  Control cells (train): {train_ctrl_mask.sum()}")

    return {
        "train_log1p":     train_log1p.astype(np.float32),
        "train_counts":    train_counts.astype(np.float32),
        "train_genes":     train_genes,
        "train_ctrl_mask": train_ctrl_mask,
        "val_log1p":       val_log1p.astype(np.float32),
        "val_counts":      val_counts.astype(np.float32),
        "val_genes":       val_genes,
        "val_ctrl_mask":   val_ctrl_mask,
        "gene_names":      selected_genes,
        "n_genes":         len(selected_genes),
        "ctrl_mean_log1p": ctrl_mean_log1p.astype(np.float32),
    }
