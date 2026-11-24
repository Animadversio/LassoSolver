#!/bin/bash
# gpu-prox round 2: matched-sparsity R2 comparison (scripts/dev_prox_sparsity.py).
#   sbatch -A binxuwang_lab -p gpu_h200 slurm/prox_sparsity.sh [--tier unit-100k]
#SBATCH -J prox-sparsity
#SBATCH -t 0-04:00
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
nvidia-smi -L
/n/home12/binxuwang/.conda/envs/torch2/bin/python -u scripts/dev_prox_sparsity.py "$@"
