#!/bin/bash
#SBATCH --job-name=geneflow-embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu          # ← change to your cluster's CPU partition

# ── adjust these two lines to match your cluster's module names ──
module load python/3.11
# module load anaconda/2023        # uncomment if using conda

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Extract perturbation gene names from the split data, then embed.
GENES=$(python3 -c "
from perturbation_fm.data.preprocess import preprocess
import numpy as np
d = preprocess('data/training.h5ad', 'data/validation.h5ad')
mask = ~d['train_ctrl_mask']
genes = sorted(set(np.asarray(d['train_genes'])[mask].tolist() +
                   np.asarray(d['val_genes'])[~d['val_ctrl_mask']].tolist()))
print(' '.join(genes))
")

mkdir -p embeddings
python -m perturbation_fm.embeddings.esm2_embed \
    --genes $GENES \
    --out   embeddings/esm2_embeddings.pt \
    --device cpu
