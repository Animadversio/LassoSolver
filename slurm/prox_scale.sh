#!/bin/bash
# gpu-prox scaling/ablation sweep. Usage:
#   sbatch -A binxuwang_lab -p gpu_h200 slurm/prox_scale.sh <dev_prox_scale.py args...>
#SBATCH -J prox-scale
#SBATCH -t 0-04:00
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
nvidia-smi -L
/n/home12/binxuwang/.conda/envs/torch2/bin/python -u scripts/dev_prox_scale.py "$@"
