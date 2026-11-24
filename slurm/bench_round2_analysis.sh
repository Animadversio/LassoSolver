#!/bin/bash
# Round-2 analysis runs (spec section 9): torch_ws on GPU + RidgeCV, with --analysis.
#   sbatch slurm/bench_round2_analysis.sh
#SBATCH -J r2-analysis
#SBATCH -p gpu_requeue
#SBATCH --constraint=h200
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=220G
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
for T in unit-100k unit-100k-raw unit-800k; do
  for S in torch_ws cpu_ridgecv; do
    echo "=== $S $T"; date
    $PY scripts/run_bench.py --solver $S --tier $T --task CV --analysis --tag round2-analysis
  done
done
date
