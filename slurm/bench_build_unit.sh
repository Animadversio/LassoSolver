#!/bin/bash
#SBATCH -J build-unit
#SBATCH -p gpu_test
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 2:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name --format=csv,noheader
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/build_unit_tiers.py "$@"
