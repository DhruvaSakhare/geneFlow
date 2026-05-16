"""Pre-compute ESM-2 (35M) gene embeddings and save to disk.

Usage:
    python -m perturbation_fm.embeddings.esm2_embed \
        --genes BRCA1 TP53 MYC \
        --out embeddings/esm2_embeddings.pt

Requires:
    pip install fair-esm requests
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch


_UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true"
    "&format=fasta&size=1"
)
_ESM_MODEL_NAME = "esm2_t12_35M_UR50D"
_EMB_DIM = 480  # mean-pool dim for the 35M model


# ---------------------------------------------------------------------------
# UniProt helpers
# ---------------------------------------------------------------------------

def _fetch_sequence(gene_name: str) -> Optional[str]:
    """Return canonical human protein sequence for *gene_name*, or None."""
    url = _UNIPROT_URL.format(gene=gene_name)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARNING] UniProt request failed for {gene_name}: {exc}")
        return None

    fasta = resp.text.strip()
    if not fasta:
        print(f"[WARNING] No UniProt entry found for {gene_name}.")
        return None

    # FASTA: first line is header, remainder is sequence (possibly multi-line).
    lines = fasta.splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq if seq else None


# ---------------------------------------------------------------------------
# ESM-2 inference
# ---------------------------------------------------------------------------

def _load_esm_model(device: torch.device):
    """Load ESM-2 35M model and alphabet."""
    import esm  # noqa: PLC0415  (deferred import — optional dependency)

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


@torch.no_grad()
def _embed_sequence(
    sequence: str,
    gene_name: str,
    model,
    batch_converter,
    device: torch.device,
) -> torch.Tensor:
    """Return mean-pooled ESM-2 embedding, shape (480,)."""
    data = [(gene_name, sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    results = model(tokens, repr_layers=[12], return_contacts=False)
    # token_representations: (1, seq_len+2, 480) — includes BOS and EOS tokens
    token_repr = results["representations"][12]
    # Mean pool over sequence positions (exclude BOS at index 0 and EOS at -1).
    mean_emb = token_repr[0, 1:-1].mean(dim=0).cpu()  # (480,)
    return mean_emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_embeddings(
    gene_names: List[str],
    out_path: str | Path,
    device_str: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Fetch UniProt sequences, embed with ESM-2, and save to *out_path*.

    Returns the embedding dict {gene_name: tensor(480,)}.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"Loading ESM-2 ({_ESM_MODEL_NAME}) on {device} …")
    model, _, batch_converter = _load_esm_model(device)

    embeddings: Dict[str, torch.Tensor] = {}
    zero_vec = torch.zeros(_EMB_DIM, dtype=torch.float32)

    for i, gene in enumerate(gene_names):
        print(f"[{i+1}/{len(gene_names)}] {gene}", end=" … ")
        seq = _fetch_sequence(gene)
        if seq is None:
            print("MISSING — using zero vector.")
            embeddings[gene] = zero_vec.clone()
            continue

        try:
            emb = _embed_sequence(seq, gene, model, batch_converter, device)
            embeddings[gene] = emb
            print(f"OK  (seq len {len(seq)})")
        except Exception as exc:
            print(f"ERROR ({exc}) — using zero vector.")
            embeddings[gene] = zero_vec.clone()

    torch.save(embeddings, out_path)
    print(f"\nSaved {len(embeddings)} embeddings to {out_path}.")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute ESM-2 gene embeddings.")
    parser.add_argument("--genes", nargs="+", required=True, help="Gene names to embed.")
    parser.add_argument(
        "--out",
        default="embeddings/esm2_embeddings.pt",
        help="Output path for the embedding dict.",
    )
    parser.add_argument("--device", default="cpu", help="torch device string.")
    args = parser.parse_args()

    compute_embeddings(args.genes, args.out, args.device)


if __name__ == "__main__":
    main()
