#!/bin/bash
#SBATCH -J build-synxl
#SBATCH -p serial_requeue
#SBATCH -A binxuwang_lab
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
date
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/build_tiers.py syn-xl --overwrite
date
