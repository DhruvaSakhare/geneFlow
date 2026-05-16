"""Convert GenePT .txt embeddings to a .pt file for use in training.

Usage:
    python -m perturbation_fm.embeddings.genept_convert \
        --input  path/to/149_genePT.txt \
        --output path/to/genept_embeddings.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert(input_path: str, output_path: str) -> None:
    gene_emb_map = {}
    with open(input_path) as f:
        for line in f:
            parts = line.strip().split()
            gene = parts[0]
            emb = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            gene_emb_map[gene] = emb

    dim = next(iter(gene_emb_map.values())).shape[0]
    print(f"Loaded {len(gene_emb_map)} genes, embedding dim={dim}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gene_emb_map, output_path)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to GenePT .txt file")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
