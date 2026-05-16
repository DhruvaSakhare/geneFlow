"""Evaluation metrics for perturbation prediction."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Atomic metrics
# ---------------------------------------------------------------------------

def pearson_lfc(pred_lfc: np.ndarray, true_lfc: np.ndarray) -> float:
    """Pearson correlation between two LFC vectors."""
    r, _ = pearsonr(pred_lfc, true_lfc)
    return float(r)


def top_k_deg_jaccard(
    pred_lfc: np.ndarray,
    true_lfc: np.ndarray,
    k: int = 50,
) -> float:
    """Jaccard index of the top-k genes by |LFC| between prediction and truth."""
    pred_top = set(np.argsort(np.abs(pred_lfc))[-k:].tolist())
    true_top = set(np.argsort(np.abs(true_lfc))[-k:].tolist())
    if not pred_top and not true_top:
        return 1.0
    return len(pred_top & true_top) / len(pred_top | true_top)


def knockdown_consistency(
    pred_lfc: np.ndarray,
    gene_name: str,
    gene_names_list: List[str],
) -> bool:
    """Return True if the knocked-out gene has pred_lfc < 0 (i.e. went down)."""
    if gene_name not in gene_names_list:
        return False
    idx = gene_names_list.index(gene_name)
    return bool(pred_lfc[idx] < 0.0)


def e_distance(pred_cells: np.ndarray, true_cells: np.ndarray) -> float:
    """Energy distance between two cell populations.

    Formula: 2·E[||X−Y||] − E[||X−X'||] − E[||Y−Y'||]

    Subsamples each population to at most 500 cells for speed.
    """
    rng = np.random.default_rng(0)
    n_sub = 500

    if len(pred_cells) > n_sub:
        pred_cells = pred_cells[rng.choice(len(pred_cells), n_sub, replace=False)]
    if len(true_cells) > n_sub:
        true_cells = true_cells[rng.choice(len(true_cells), n_sub, replace=False)]

    xy = cdist(pred_cells, true_cells, metric="euclidean").mean()
    xx = cdist(pred_cells, pred_cells, metric="euclidean").mean()
    yy = cdist(true_cells, true_cells, metric="euclidean").mean()

    return float(2.0 * xy - xx - yy)


def variance_correlation(pred_cells: np.ndarray, true_cells: np.ndarray) -> float:
    """Pearson correlation between per-gene variance of predicted vs. true cells."""
    pred_var = pred_cells.var(axis=0)
    true_var = true_cells.var(axis=0)
    r, _ = pearsonr(pred_var, true_var)
    return float(r)


# ---------------------------------------------------------------------------
# Per-perturbation evaluation
# ---------------------------------------------------------------------------

def evaluate_perturbation(
    model,
    control_log1p: np.ndarray,
    true_pert_log1p: np.ndarray,
    true_pert_counts: np.ndarray,
    ctrl_mean_log1p: np.ndarray,
    esm_emb: torch.Tensor,
    gene_name: str,
    gene_names_list: List[str],
    n_sample: int = 1000,
    device: str = "cpu",
) -> Dict:
    """Full evaluation pipeline for one perturbation.

    Args:
        model:            PerturbationFlowModel (in eval mode).
        control_log1p:    (N_ctrl, n_genes) all available control cells.
        true_pert_log1p:  (M, n_genes) true perturbed log1p cells.
        true_pert_counts: (M, n_genes) true raw counts.
        ctrl_mean_log1p:  (n_genes,) reference control mean.
        esm_emb:          (esm_dim,) gene embedding tensor.
        gene_name:        name of the knocked-out gene.
        gene_names_list:  ordered list of selected gene names.
        n_sample:         number of control cells to forward through the model.
        device:           torch device string.

    Returns:
        dict with keys pearson_r, jaccard_top50, knockdown_ok,
                       e_distance, variance_corr.
    """
    dev = torch.device(device)

    # a. Sample n_sample control cells.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(control_log1p), n_sample, replace=len(control_log1p) < n_sample)
    ctrl_sample = control_log1p[idx]

    # b. Predict.
    x0_tensor = torch.from_numpy(ctrl_sample).float().to(dev)
    esm_tensor = esm_emb.float().to(dev)

    model.eval()
    out = model.predict(x0_tensor, esm_tensor)

    # c. Convert sampled counts to log1p space for population-level metrics.
    counts_np = out["counts"].cpu().float().numpy()
    totals = counts_np.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    pred_log1p = np.log1p(counts_np / totals * 10_000.0)  # (n_sample, n_genes)

    # d. Predicted LFC.
    pred_lfc = pred_log1p.mean(axis=0) - ctrl_mean_log1p

    # e. True LFC.
    true_lfc = true_pert_log1p.mean(axis=0) - ctrl_mean_log1p

    return {
        "pearson_r":     pearson_lfc(pred_lfc, true_lfc),
        "jaccard_top50": top_k_deg_jaccard(pred_lfc, true_lfc, k=50),
        "knockdown_ok":  knockdown_consistency(pred_lfc, gene_name, gene_names_list),
        "e_distance":    e_distance(pred_log1p, true_pert_log1p),
        "variance_corr": variance_correlation(pred_log1p, true_pert_log1p),
    }


# ---------------------------------------------------------------------------
# Full validation sweep
# ---------------------------------------------------------------------------

def evaluate_all_perturbations(
    model,
    data_dict: Dict,
    gene_emb_map: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Dict:
    """Evaluate all validation perturbations and return aggregated results.

    Uses training control cells as x0 (same distribution the model was trained on).

    Returns:
        {
          "mean": {metric: float},
          "per_perturbation": {gene: {metric: value}},
        }
    """
    train_log1p = data_dict["train_log1p"]
    train_ctrl_mask = data_dict["train_ctrl_mask"]
    control_log1p = train_log1p[train_ctrl_mask]

    val_log1p = data_dict["val_log1p"]
    val_counts = data_dict["val_counts"]
    val_genes = np.asarray(data_dict["val_genes"])
    val_ctrl_mask = data_dict["val_ctrl_mask"]
    ctrl_mean = data_dict["ctrl_mean_log1p"]
    gene_names_list = data_dict["gene_names"]

    val_perturbations = np.unique(val_genes[~val_ctrl_mask])

    per_pert: Dict[str, Dict] = {}
    for gene in val_perturbations:
        mask = (~val_ctrl_mask) & (val_genes == gene)
        true_pert_log1p = val_log1p[mask]
        true_pert_counts = val_counts[mask]

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
            device=device,
        )
        per_pert[gene] = metrics

    # Aggregate means (numeric metrics only).
    numeric_keys = ["pearson_r", "jaccard_top50", "e_distance", "variance_corr"]
    means = {k: float(np.mean([p[k] for p in per_pert.values()])) for k in numeric_keys}
    means["knockdown_ok_rate"] = float(
        np.mean([float(p["knockdown_ok"]) for p in per_pert.values()])
    )

    # Print summary table.
    header = f"{'Perturbation':<20} {'Pearson r':>10} {'Jaccard@50':>12} {'E-dist':>10} {'Var-corr':>10} {'KD-ok':>7}"
    print("\n" + header)
    print("-" * len(header))
    for gene, m in sorted(per_pert.items()):
        print(
            f"{gene:<20} {m['pearson_r']:>10.4f} {m['jaccard_top50']:>12.4f} "
            f"{m['e_distance']:>10.4f} {m['variance_corr']:>10.4f} {str(m['knockdown_ok']):>7}"
        )
    print("-" * len(header))
    print(
        f"{'MEAN':<20} {means['pearson_r']:>10.4f} {means['jaccard_top50']:>12.4f} "
        f"{means['e_distance']:>10.4f} {means['variance_corr']:>10.4f} "
        f"{means['knockdown_ok_rate']:>7.4f}"
    )

    return {"mean": means, "per_perturbation": per_pert}
