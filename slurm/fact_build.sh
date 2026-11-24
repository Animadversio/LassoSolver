#!/bin/bash
# factorized-readout: extract resnet50_robust layer3 maps + build tier unit-200k-l3.
#   sbatch slurm/fact_build.sh [--block 5]
#SBATCH -J fact-build
#SBATCH -p gpu_test
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH -t 1:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name --format=csv,noheader
/n/home12/binxuwang/.conda/envs/torch2/bin/python -u scripts/dev_fact_build_l3.py "$@"
