"""Evaluation metrics for perturbation prediction."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Atomic metrics
# ---------------------------------------------------------------------------

def pearson_lfc(pred_lfc: np.ndarray, true_lfc: np.ndarray) -> float:
    """Pearson correlation between two LFC vectors."""
    r, _ = pearsonr(pred_lfc, true_lfc)
    return float(r)


def weighted_cosine_lfc(
    pred_lfc: np.ndarray,
    true_lfc: np.ndarray,
    gate_low: float = 0.0,
    gate_high: float = 0.2,
    eps: float = 1e-12,
) -> float:
    """Weighted cosine similarity with smoothstep gating on gene activity.

    Only genes where max(|pred|, |true|) > gate_low contribute, with full
    weight above gate_high. Focuses on DE genes, ignoring the ~96% near-zero.
    """
    x = np.maximum(np.abs(pred_lfc), np.abs(true_lfc))
    t = np.clip((x - gate_low) / (gate_high - gate_low), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)  # smoothstep
    w2 = w * w

    num = float(np.sum(w2 * pred_lfc * true_lfc))
    den = float(np.sqrt(np.sum(w2 * pred_lfc ** 2)) * np.sqrt(np.sum(w2 * true_lfc ** 2)))
    if den < eps:
        return 0.0
    return float(num / den)


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


def e_distance(pred_cells: np.ndarray, true_cells: np.ndarray, n_sub: int = 200) -> float:
    """Energy distance between two cell populations.

    Formula: 2·E[||X−Y||] − E[||X−X'||] − E[||Y−Y'||]

    Uses torch.cdist on GPU/CPU for speed. Subsamples to n_sub cells each.
    """
    rng = np.random.default_rng(0)

    if len(pred_cells) > n_sub:
        pred_cells = pred_cells[rng.choice(len(pred_cells), n_sub, replace=False)]
    if len(true_cells) > n_sub:
        true_cells = true_cells[rng.choice(len(true_cells), n_sub, replace=False)]

    p = torch.from_numpy(pred_cells).float()
    t = torch.from_numpy(true_cells).float()

    xy = torch.cdist(p, t).mean().item()
    xx = torch.cdist(p, p).mean().item()
    yy = torch.cdist(t, t).mean().item()

    return float(2.0 * xy - xx - yy)


def variance_correlation(pred_cells: np.ndarray, true_cells: np.ndarray) -> float:
    """Pearson correlation between per-gene variance of predicted vs. true cells."""
    pred_var = pred_cells.var(axis=0)
    true_var = true_cells.var(axis=0)
    r, _ = pearsonr(pred_var, true_var)
    return float(r)


def perturbation_discrimination_score(
    pred_lfcs: Dict[str, np.ndarray],
    true_lfcs: Dict[str, np.ndarray],
) -> float:
    """Can the model distinguish between perturbations?

    For each true perturbation, rank all predicted LFC vectors by similarity
    (negative L2 distance). Score = 1 - mean(normalized_rank), where rank 0
    means the matching prediction was closest. 1.0 = perfect discrimination,
    0.5 = random.
    """
    perts = sorted(set(pred_lfcs.keys()) & set(true_lfcs.keys()))
    if len(perts) < 2:
        return float("nan")

    pred_mat = np.stack([pred_lfcs[p] for p in perts])  # (P, n_genes)
    true_mat = np.stack([true_lfcs[p] for p in perts])  # (P, n_genes)

    # Pairwise L2 distances between each true vector and all predictions.
    dists = np.linalg.norm(true_mat[:, None, :] - pred_mat[None, :, :], axis=2)
    # For each row i, where does column i (the matching prediction) rank?
    ranks = np.argsort(dists, axis=1)
    normalized_ranks = []
    for i in range(len(perts)):
        rank_of_match = int(np.where(ranks[i] == i)[0][0])
        normalized_ranks.append(rank_of_match / (len(perts) - 1))
    return float(1.0 - np.mean(normalized_ranks))


def sliced_wasserstein2(
    pred_cells: np.ndarray,
    true_cells: np.ndarray,
    n_projections: int = 50,
    n_sub: int = 500,
    seed: int = 0,
) -> float:
    """Sliced Wasserstein-2 distance between two cell populations.

    Projects onto random 1D directions, computes W2 on each (sorted L2 distance),
    averages across projections. Natural metric for flow matching since FM is
    connected to optimal transport.
    """
    rng = np.random.default_rng(seed)

    if len(pred_cells) > n_sub:
        pred_cells = pred_cells[rng.choice(len(pred_cells), n_sub, replace=False)]
    if len(true_cells) > n_sub:
        true_cells = true_cells[rng.choice(len(true_cells), n_sub, replace=False)]

    d = pred_cells.shape[1]
    projections = rng.standard_normal(size=(n_projections, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    p_proj = pred_cells @ projections.T  # (n_pred, n_projections)
    t_proj = true_cells @ projections.T  # (n_true, n_projections)

    p_sorted = np.sort(p_proj, axis=0)
    t_sorted = np.sort(t_proj, axis=0)

    # Resample to common length if needed.
    n = min(p_sorted.shape[0], t_sorted.shape[0])
    if p_sorted.shape[0] != n:
        idx = np.linspace(0, p_sorted.shape[0] - 1, n).astype(int)
        p_sorted = p_sorted[idx]
    if t_sorted.shape[0] != n:
        idx = np.linspace(0, t_sorted.shape[0] - 1, n).astype(int)
        t_sorted = t_sorted[idx]

    w2_per_proj = np.mean((p_sorted - t_sorted) ** 2, axis=0)
    return float(np.sqrt(w2_per_proj.mean()))


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

    # c. ODE endpoint (already includes baseline_lfc added back inside model.predict).
    # Use directly for both LFC and population metrics — no NB head transformation.
    pred_log1p = out["x1_log1p"].cpu().float().numpy()

    # d. Predicted LFC from ODE output.
    pred_lfc = pred_log1p.mean(axis=0) - ctrl_mean_log1p

    # e. True LFC.
    true_lfc = true_pert_log1p.mean(axis=0) - ctrl_mean_log1p

    return {
        "pearson_r":      pearson_lfc(pred_lfc, true_lfc),
        "weighted_cos":   weighted_cosine_lfc(pred_lfc, true_lfc),
        "jaccard_top50":  top_k_deg_jaccard(pred_lfc, true_lfc, k=50),
        "knockdown_ok":   knockdown_consistency(pred_lfc, gene_name, gene_names_list),
        "e_distance":     e_distance(pred_log1p, true_pert_log1p),
        "variance_corr":  variance_correlation(pred_log1p, true_pert_log1p),
        "sliced_w2":      sliced_wasserstein2(pred_log1p, true_pert_log1p),
        "_pred_lfc":      pred_lfc,
        "_true_lfc":      true_lfc,
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
    numeric_keys = ["pearson_r", "weighted_cos", "jaccard_top50", "e_distance", "variance_corr", "sliced_w2"]
    means = {k: float(np.mean([p[k] for p in per_pert.values()])) for k in numeric_keys}
    means["knockdown_ok_rate"] = float(
        np.mean([float(p["knockdown_ok"]) for p in per_pert.values()])
    )

    # Perturbation Discrimination Score (cross-perturbation metric).
    pred_lfcs = {g: m["_pred_lfc"] for g, m in per_pert.items()}
    true_lfcs = {g: m["_true_lfc"] for g, m in per_pert.items()}
    means["pds"] = perturbation_discrimination_score(pred_lfcs, true_lfcs)

    # Print summary table — cell distribution metrics emphasized.
    header = (
        f"{'Perturbation':<20} {'Pearson r':>10} {'Wtd-cos':>10} "
        f"{'E-dist':>10} {'SW2':>10} {'Var-corr':>10} {'KD-ok':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for gene, m in sorted(per_pert.items()):
        print(
            f"{gene:<20} {m['pearson_r']:>10.4f} {m['weighted_cos']:>10.4f} "
            f"{m['e_distance']:>10.4f} {m['sliced_w2']:>10.4f} "
            f"{m['variance_corr']:>10.4f} {str(m['knockdown_ok']):>7}"
        )
    print("-" * len(header))
    print(
        f"{'MEAN':<20} {means['pearson_r']:>10.4f} {means['weighted_cos']:>10.4f} "
        f"{means['e_distance']:>10.4f} {means['sliced_w2']:>10.4f} "
        f"{means['variance_corr']:>10.4f} {means['knockdown_ok_rate']:>7.4f}"
    )
    print(f"PDS (Perturbation Discrimination Score): {means['pds']:.4f}  "
          f"[1.0 = perfect, 0.5 = random]")

    return {"mean": means, "per_perturbation": per_pert}
