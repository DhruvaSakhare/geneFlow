#!/bin/bash
#SBATCH --job-name=geneflow-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1             # ← 1 GPU; change type e.g. gpu:a100:1
#SBATCH --partition=gpu          # ← change to your cluster's GPU partition

# ── adjust to match your cluster's modules ──
module load python/3.11
module load cuda/12.1

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

mkdir -p logs checkpoints

python -m perturbation_fm.training.train \
    --config perturbation_fm/configs/default.yaml
