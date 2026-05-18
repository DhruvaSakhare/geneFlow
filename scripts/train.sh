#!/bin/sh
### DTU GBAR LSF submission script for FlowNet training on a single A100.
### Submit with:   bsub < scripts/train.sh
### Check status:  bstat -u $USER     |   bjobs
### Kill:          bkill <jobid>

#BSUB -J geneflow-train
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -N

# ---------- Modules ----------
module load cuda/12.1

# ---------- Activate conda env ----------
source /work3/s225191/miniforge3/bin/activate geneflow

cd $LS_SUBCWD
mkdir -p logs checkpoints

# ---------- Run ----------
echo "Job ID: $LSB_JOBID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
nvidia-smi

python -m perturbation_fm.training.train \
    --config perturbation_fm/configs/default.yaml

echo "End: $(date)"
