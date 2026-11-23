#!/bin/bash
#SBATCH -J bench-ref
#SBATCH -p serial_requeue
#SBATCH -A binxuwang_lab
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
# Usage: sbatch slurm/bench_reference.sh --tier unit-100k [build_reference.py args]
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
lscpu | grep "Model name"; nproc; date
/n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/build_reference.py "$@"
date
