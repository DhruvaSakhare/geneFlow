"""PyTorch Dataset and DataLoader for perturbation flow matching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PerturbationDataset(Dataset):
    """Returns (x0_log1p, x1_log1p, x1_counts, esm_emb) tuples.

    Each index maps to a single perturbed cell (x1).  A random control cell
    (x0) is sampled on the fly so every epoch sees different (x0, x1) pairs.
    """

    def __init__(
        self,
        log1p_data: np.ndarray,
        counts_data: np.ndarray,
        gene_targets: List[str],
        ctrl_mask: np.ndarray,
        gene_emb_map: Dict[str, torch.Tensor],
    ) -> None:
        self.log1p = log1p_data
        self.counts = counts_data
        self.gene_targets = np.asarray(gene_targets)
        self.ctrl_mask = np.asarray(ctrl_mask, dtype=bool)
        self.gene_emb_map = gene_emb_map

        self.pert_indices: np.ndarray = np.where(~self.ctrl_mask)[0]
        self.ctrl_indices: np.ndarray = np.where(self.ctrl_mask)[0]

        # Infer ESM embedding dimension from any stored entry.
        self._esm_dim: int = next(iter(gene_emb_map.values())).shape[0]

    def __len__(self) -> int:
        return len(self.pert_indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pert_cell_idx = self.pert_indices[idx]

        x1_log1p = torch.from_numpy(self.log1p[pert_cell_idx]).float()
        x1_counts = torch.from_numpy(self.counts[pert_cell_idx]).float()

        # Sample a random control cell as x0.
        ctrl_idx = int(np.random.choice(self.ctrl_indices))
        x0_log1p = torch.from_numpy(self.log1p[ctrl_idx]).float()

        # ESM embedding for the perturbed gene.
        gene = self.gene_targets[pert_cell_idx]
        esm_emb = self.gene_emb_map.get(gene)
        if esm_emb is None:
            esm_emb = torch.zeros(self._esm_dim, dtype=torch.float32)

        return x0_log1p, x1_log1p, x1_counts, esm_emb.float()


def get_dataloader(
    dataset: PerturbationDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
