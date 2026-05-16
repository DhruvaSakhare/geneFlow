"""Add ESM-2 embeddings for missing genes to an existing embedding file.

Loads the existing .pt file, identifies missing val perturbation genes,
fetches/embeds them, and saves the merged file.

Usage:
    python -m perturbation_fm.embeddings.add_val_embeddings \
        --existing /work3/s225191/geneFlow/embeddings/esm2_embeddings.pt \
        --val_path /work3/s225191/geneFlow/data/validation.h5ad \
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

from perturbation_fm.embeddings.esm2_embed import compute_embeddings


def _get_val_perturbation_genes(val_path: str) -> List[str]:
    import scanpy as sc
    adata = sc.read_h5ad(val_path)
    genes = adata.obs["target_gene"].unique().tolist()
    return [g for g in genes if g != "non-targeting"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True, help="Path to existing .pt embedding file")
    parser.add_argument("--val_path", required=True, help="Path to validation.h5ad")
    parser.add_argument("--device",   default="cpu")
    args = parser.parse_args()

    existing_path = Path(args.existing)
    existing = torch.load(existing_path, map_location="cpu")
    print(f"Loaded {len(existing)} existing embeddings from {existing_path}")

    val_genes = _get_val_perturbation_genes(args.val_path)
    print(f"Val perturbation genes ({len(val_genes)}): {val_genes}")

    missing = [g for g in val_genes if g not in existing]
    print(f"Missing from embedding file: {len(missing)} genes")

    if not missing:
        print("Nothing to do.")
        return

    # Embed missing genes (writes to a temp file then merges)
    tmp_path = existing_path.with_suffix(".tmp.pt")
    new_embs = compute_embeddings(missing, tmp_path, device_str=args.device)
    tmp_path.unlink(missing_ok=True)

    # Merge
    existing.update(new_embs)
    torch.save(existing, existing_path)
    print(f"\nMerged file saved: {len(existing)} total embeddings → {existing_path}")


if __name__ == "__main__":
    main()
