"""PyTorch Dataset and DataLoader for perturbation flow matching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PerturbationDataset(Dataset):
    """Returns (x0_log1p, x1_log1p, esm_emb) tuples.

    Each index maps to a single perturbed cell (x1).  A random control cell
    (x0) is sampled on the fly so every epoch sees different (x0, x1) pairs.
    """

    def __init__(
        self,
        log1p_data: np.ndarray,
        gene_targets: List[str],
        ctrl_mask: np.ndarray,
        gene_emb_map: Dict[str, torch.Tensor],
    ) -> None:
        self.log1p = log1p_data
        self.gene_targets = np.asarray(gene_targets)
        self.ctrl_mask = np.asarray(ctrl_mask, dtype=bool)
        self.gene_emb_map = gene_emb_map

        self.pert_indices: np.ndarray = np.where(~self.ctrl_mask)[0]
        self.ctrl_indices: np.ndarray = np.where(self.ctrl_mask)[0]

        # Per-perturbation local index (into pert_indices, not raw data).
        self.pert_to_local: Dict[str, List[int]] = {}
        for local_idx, global_idx in enumerate(self.pert_indices):
            gene = self.gene_targets[global_idx]
            self.pert_to_local.setdefault(gene, []).append(local_idx)
        self.unique_perts: List[str] = sorted(self.pert_to_local.keys())

        self._esm_dim: int = next(iter(gene_emb_map.values())).shape[0]

    def __len__(self) -> int:
        return len(self.pert_indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pert_cell_idx = self.pert_indices[idx]

        x1_log1p = torch.from_numpy(self.log1p[pert_cell_idx]).float()

        ctrl_idx = int(np.random.choice(self.ctrl_indices))
        x0_log1p = torch.from_numpy(self.log1p[ctrl_idx]).float()

        gene = self.gene_targets[pert_cell_idx]
        esm_emb = self.gene_emb_map.get(gene)
        if esm_emb is None:
            esm_emb = torch.zeros(self._esm_dim, dtype=torch.float32)

        return x0_log1p, x1_log1p, esm_emb.float()


class BalancedPerturbationBatchSampler:
    """Yields batches with P randomly chosen perturbations × K cells each.

    This gives OT pairing K cells per perturbation (not ~2), making
    within-perturbation OT meaningful.
    """

    def __init__(
        self,
        dataset: PerturbationDataset,
        n_perts_per_batch: int = 16,
        n_cells_per_pert: int = 16,
        n_batches: int = 500,
    ) -> None:
        self.P = n_perts_per_batch
        self.K = n_cells_per_pert
        self.n_batches = n_batches
        self.pert_to_local = dataset.pert_to_local
        self.unique_perts = dataset.unique_perts
        assert len(self.unique_perts) >= n_perts_per_batch, (
            f"Only {len(self.unique_perts)} perturbations available, "
            f"need at least n_perts_per_batch={n_perts_per_batch}"
        )

    def __iter__(self):
        for _ in range(self.n_batches):
            perts = np.random.choice(self.unique_perts, self.P, replace=False)
            batch: List[int] = []
            for pert in perts:
                indices = self.pert_to_local[pert]
                chosen = np.random.choice(
                    indices, self.K, replace=len(indices) < self.K
                )
                batch.extend(chosen.tolist())
            yield batch

    def __len__(self) -> int:
        return self.n_batches


def get_dataloader(
    dataset: PerturbationDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    balanced: bool = False,
    n_perts_per_batch: int = 16,
    n_cells_per_pert: int = 16,
) -> DataLoader:
    if balanced:
        n_batches = len(dataset) // (n_perts_per_batch * n_cells_per_pert)
        sampler = BalancedPerturbationBatchSampler(
            dataset,
            n_perts_per_batch=n_perts_per_batch,
            n_cells_per_pert=n_cells_per_pert,
            n_batches=n_batches,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
