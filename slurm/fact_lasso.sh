#!/bin/bash
# factorized-readout: unit-basis Lasso coefficients, path, 5-member stability, spatial concentration.
#   sbatch -p gpu_h200 slurm/fact_lasso.sh --tier unit-100k
#SBATCH -J fact-lasso
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH -t 0-03:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
nvidia-smi -L
/n/home12/binxuwang/.conda/envs/torch2/bin/python -u scripts/dev_fact_lasso.py "$@"
