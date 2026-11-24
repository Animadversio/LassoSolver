#!/bin/bash
# factorized-readout round 2 (splits / stability / ablations / timing).
#   sbatch -p gpu_h200 slurm/fact_round2.sh --tier unit-100k --parts splits,stab,ablate
#SBATCH -J fact-r2
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=150G
#SBATCH -t 0-06:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
nvidia-smi -L
/n/home12/binxuwang/.conda/envs/torch2/bin/python -u -m lassobench.factorized.round2 "$@"
