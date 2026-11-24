#!/bin/bash
#SBATCH -J ws-celer
#SBATCH -A binxuwang_lab
#SBATCH -p shared
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/ws-celer-%j.out
# celer yardstick on the datasets saved by the gpu-workingset scaling grid (32 targets, 32 processes).
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
C=/n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/cache/ws_syn
for cfg in "$@"; do
  set -- $cfg
  timeout 14400 $PY scripts/dev_ws_celer_saved.py $C/$1 $2 32 32
done
