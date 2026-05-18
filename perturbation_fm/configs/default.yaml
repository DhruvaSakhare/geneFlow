# PerturbationFlowModel — default hyperparameters

# Model architecture
n_genes:    null        # filled automatically after preprocessing
esm_dim:    480         # ESM-2 35M output dimension
hidden_dim: 512
num_layers: 8
dropout:    0.2
lambda_lfc: 0.3         # weight on the per-perturbation mean-LFC loss

# Optimisation
batch_size:       256
n_perts_per_batch: 16   # perturbations per batch (for balanced sampler + per-pert OT)
n_cells_per_pert:  16   # cells per perturbation  (16 × 16 = 256 total)
lr:           1.0e-4
weight_decay: 1.0e-4
n_epochs:     100
patience:     30        # early-stopping patience on val FM loss
grad_clip:    1.0

# Inference / evaluation
n_inference_cells: 1000   # control cells sampled per perturbation at eval
ode_steps:         50     # RK4 integration steps

# Reproducibility
seed: 42

# Paths (relative to repo root: geneFlow/)
train_path: /work3/s225191/geneFlow/data/training.h5ad
val_path:   /work3/s225191/geneFlow/data/validation.h5ad
emb_path:   /work3/s225191/geneFlow/embeddings/esm2_embeddings.pt

save_dir:   /work3/s225191/geneFlow/checkpoints/
