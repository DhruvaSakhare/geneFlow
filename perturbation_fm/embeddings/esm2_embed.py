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
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch


_UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true"
    "&format=fasta&size=1"
)
_ESM_MODEL_NAME = "esm2_t12_35M_UR50D"
_EMB_DIM = 480


# ---------------------------------------------------------------------------
# UniProt — fetch all sequences in parallel
# ---------------------------------------------------------------------------

def _fetch_sequence(gene_name: str) -> Tuple[str, Optional[str]]:
    """Return (gene_name, sequence_or_None)."""
    url = _UNIPROT_URL.format(gene=gene_name)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARNING] UniProt request failed for {gene_name}: {exc}")
        return gene_name, None

    fasta = resp.text.strip()
    if not fasta:
        print(f"[WARNING] No UniProt entry found for {gene_name}.")
        return gene_name, None

    lines = fasta.splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return gene_name, (seq if seq else None)


def _fetch_all_sequences(
    gene_names: List[str],
    n_workers: int = 8,
) -> Dict[str, Optional[str]]:
    """Fetch UniProt sequences for all genes in parallel."""
    print(f"Fetching {len(gene_names)} sequences from UniProt "
          f"({n_workers} parallel workers) …")
    results: Dict[str, Optional[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fetch_sequence, g): g for g in gene_names}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            gene, seq = fut.result()
            results[gene] = seq
            status = f"OK (len {len(seq)})" if seq else "MISSING"
            print(f"  [{i}/{len(gene_names)}] {gene}: {status}")
    return results


# ---------------------------------------------------------------------------
# ESM-2 — embed in sorted batches
# ---------------------------------------------------------------------------

def _load_esm_model(device: torch.device):
    import esm  # noqa: PLC0415
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.to(device).eval()
    return model, alphabet.get_batch_converter()


@torch.no_grad()
def _embed_batch(
    batch_data: List[Tuple[str, str]],
    model,
    batch_converter,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Embed a batch of (gene_name, sequence) pairs, return {gene: tensor(480,)}."""
    _, _, tokens = batch_converter(batch_data)
    tokens = tokens.to(device)

    results = model(tokens, repr_layers=[12], return_contacts=False)
    token_repr = results["representations"][12]  # (B, max_len+2, 480)

    embeddings = {}
    for i, (gene, seq) in enumerate(batch_data):
        # Exclude BOS (index 0) and EOS (index seq_len+1).
        seq_len = len(seq)
        embeddings[gene] = token_repr[i, 1 : seq_len + 1].mean(dim=0).cpu()
    return embeddings


def _embed_all(
    sequences: Dict[str, str],
    model,
    batch_converter,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Embed all sequences in length-sorted batches to minimise padding waste."""
    # Sort by sequence length so sequences in each batch have similar lengths.
    sorted_items = sorted(sequences.items(), key=lambda kv: len(kv[1]))

    embeddings: Dict[str, torch.Tensor] = {}
    n = len(sorted_items)
    for start in range(0, n, batch_size):
        batch = sorted_items[start : start + batch_size]
        end = min(start + batch_size, n)
        print(f"  Embedding genes {start+1}–{end}/{n} …")
        batch_embs = _embed_batch(batch, model, batch_converter, device)
        embeddings.update(batch_embs)
    return embeddings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_embeddings(
    gene_names: List[str],
    out_path: str | Path,
    device_str: str = "cpu",
    batch_size: int = 16,
    n_fetch_workers: int = 8,
) -> Dict[str, torch.Tensor]:
    """Fetch UniProt sequences, embed with ESM-2, save to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetch all sequences in parallel.
    raw = _fetch_all_sequences(gene_names, n_workers=n_fetch_workers)
    good = {g: s for g, s in raw.items() if s is not None}
    missing = [g for g, s in raw.items() if s is None]
    print(f"\n{len(good)} sequences fetched, {len(missing)} missing.")

    # 2. Load model.
    device = torch.device(device_str)
    print(f"Loading ESM-2 ({_ESM_MODEL_NAME}) on {device} …")
    model, batch_converter = _load_esm_model(device)

    # 3. Embed in batches.
    embeddings = _embed_all(good, model, batch_converter, device, batch_size)

    # 4. Fill missing genes with zero vectors.
    zero = torch.zeros(_EMB_DIM, dtype=torch.float32)
    for gene in missing:
        embeddings[gene] = zero.clone()

    torch.save(embeddings, out_path)
    print(f"\nSaved {len(embeddings)} embeddings → {out_path}")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute ESM-2 gene embeddings.")
    parser.add_argument("--genes",   nargs="+", required=True)
    parser.add_argument("--out",     default="embeddings/esm2_embeddings.pt")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--batch-size",      type=int, default=16)
    parser.add_argument("--fetch-workers",   type=int, default=8)
    args = parser.parse_args()

    compute_embeddings(
        args.genes,
        args.out,
        device_str=args.device,
        batch_size=args.batch_size,
        n_fetch_workers=args.fetch_workers,
    )


if __name__ == "__main__":
    main()
