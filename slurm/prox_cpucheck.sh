#!/bin/bash
# CPU-only quick logic check of gpu-prox solvers (device=cpu).
#SBATCH -J prox-cpu
#SBATCH -t 0-00:30
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
$PY -u scripts/dev_prox_check.py --n 100 --p 500 --T 2 --A 5 --device cpu --opts '{"dtype":"float64","max_iter":50000}'
$PY -u scripts/dev_prox_check.py --n 100 --p 500 --T 2 --A 5 --device cpu --solver torch_proxgrad --opts '{"dtype":"float32","epochs":3000,"lr":1e-2}'
