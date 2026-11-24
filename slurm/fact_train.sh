#!/bin/bash
# factorized-readout: train readouts on a tier.
#   sbatch -p gpu_h200 slurm/fact_train.sh --tier unit-100k --model gauss_mse gauss_lin [--gammas ...]
#SBATCH -J fact-train
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH -t 0-04:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
nvidia-smi -L
/n/home12/binxuwang/.conda/envs/torch2/bin/python -u -m lassobench.factorized.run "$@"
