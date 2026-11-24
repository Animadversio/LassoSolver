#!/bin/bash
# factorized-readout: harness Ridge / Lasso comparison rows on a tier (default unit-200k-l3).
#   sbatch -p gpu_h200 slurm/fact_harness.sh [tier]
#SBATCH -J fact-harness
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH -t 0-01:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
TIER=${1:-unit-200k-l3}
nvidia-smi -L
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
$PY -u scripts/run_bench.py --solver cpu_ridgecv --tier $TIER --task CV --tag factorized-readout
$PY -u scripts/run_bench.py --solver torch_ws --tier $TIER --task CV --solver-args tol=1e-4 --tag factorized-readout
