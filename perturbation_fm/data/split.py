"""Split a single .h5ad file into train / val / test by perturbation identity.

Splitting by cell would leak: the model would be tested on perturbations it
already saw. Instead we hold out entire perturbations so evaluation measures
generalisation to unseen knock-outs.

Control cells are included in every split — they are needed as x0 at both
training and evaluation time.

Usage (from geneFlow/):
    python -m perturbation_fm.data.split \
        --input  data/training.h5ad \
        --outdir data/ \
        --train  0.80 \
        --val    0.10 \
        --test   0.10 \
        --seed   42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData


def split_by_perturbation(
    adata: AnnData,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[AnnData, AnnData, AnnData]:
    """Split adata into (train, val, test) by held-out perturbations.

    Control cells (obs["control"] == True) are copied into every split so
    the model always has a pool of x0 cells to draw from.

    Args:
        adata:       Full AnnData with obs["gene_target"] and obs["control"].
        train_frac:  Fraction of perturbations for training.
        val_frac:    Fraction for validation (remainder goes to test).
        seed:        Random seed for reproducibility.

    Returns:
        (adata_train, adata_val, adata_test)
    """
    assert abs(train_frac + val_frac - 1.0) <= 0.5, \
        "train_frac + val_frac must be <= 1.0"

    ctrl_mask = np.asarray(adata.obs["control"].values, dtype=bool)
    ctrl_adata = adata[ctrl_mask].copy()

    # All unique perturbation genes (excluding control sentinel values).
    pert_genes = np.asarray(
        sorted(set(adata.obs.loc[~ctrl_mask, "gene_target"].tolist()))
    )
    n_perts = len(pert_genes)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pert_genes)

    n_train = round(n_perts * train_frac)
    n_val   = round(n_perts * val_frac)
    # test gets whatever is left
    train_genes = set(shuffled[:n_train].tolist())
    val_genes   = set(shuffled[n_train : n_train + n_val].tolist())
    test_genes  = set(shuffled[n_train + n_val :].tolist())

    print(f"Perturbation split — train: {len(train_genes)}, "
          f"val: {len(val_genes)}, test: {len(test_genes)}")

    def _subset(gene_set: set) -> AnnData:
        pert_mask = np.asarray(
            [g in gene_set for g in adata.obs["gene_target"].values]
        ) & (~ctrl_mask)
        # Concatenate matching perturbed cells + all control cells.
        return sc.concat([adata[pert_mask], ctrl_adata], join="outer")

    adata_train = _subset(train_genes)
    adata_val   = _subset(val_genes)
    adata_test  = _subset(test_genes)

    for name, ad in [("train", adata_train), ("val", adata_val), ("test", adata_test)]:
        n_ctrl = ad.obs["control"].sum()
        n_pert = (~ad.obs["control"].astype(bool)).sum()
        print(f"  {name:5s}: {n_pert:6d} perturbed cells + {n_ctrl:6d} control cells "
              f"= {len(ad):6d} total")

    return adata_train, adata_val, adata_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Split h5ad by perturbation.")
    parser.add_argument("--input",  required=True, help="Path to the source .h5ad file.")
    parser.add_argument("--outdir", default="data/", help="Directory for output files.")
    parser.add_argument("--train",  type=float, default=0.80)
    parser.add_argument("--val",    type=float, default=0.10)
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} …")
    adata = sc.read_h5ad(args.input)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    adata_train, adata_val, adata_test = split_by_perturbation(
        adata,
        train_frac=args.train,
        val_frac=args.val,
        seed=args.seed,
    )

    for name, ad in [("training", adata_train), ("validation", adata_val), ("test", adata_test)]:
        out_path = outdir / f"{name}.h5ad"
        ad.write_h5ad(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
