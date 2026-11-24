#!/bin/bash
# gpu-prox: torch_fista through the harness on the unit-basis tiers (Task P and CV, tol 1e-4).
#   sbatch -A binxuwang_lab -p gpu_h200 slurm/prox_harness_fista.sh ; sbatch -A binxuwang_lab -p gpu slurm/prox_harness_fista.sh
#SBATCH -J prox-hfista
#SBATCH -t 0-10:00
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
nvidia-smi -L
rb() { $PY -u scripts/run_bench.py --tag gpu-prox "$@"; }
rb --solver torch_fista --tier unit-100k --task P  --solver-args tol=1e-4
rb --solver torch_fista --tier unit-100k --task P  --solver-args tol=1e-4 screen=True
rb --solver torch_fista --tier unit-100k --task CV --solver-args tol=1e-4
rb --solver torch_fista --tier unit-800k --task P  --solver-args tol=1e-4 screen=True --timeout 3000
rb --solver torch_fista --tier unit-800k --task P  --solver-args tol=1e-4 --timeout 3000
rb --solver torch_fista --tier unit-800k --task CV --solver-args tol=1e-4 screen=True --timeout 6000
