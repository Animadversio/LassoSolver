#!/bin/bash
#SBATCH -J r2-analysis
#SBATCH -p gpu_requeue
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=220G
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH -o /dev/null
# Account: pass it on the command line (sbatch -A <account> ...). Submit from the repo root: output goes to
# $LASSOBENCH_STORE/results/logs/%x-%j.out (see slurm/bench_common.sh).
# Round-2 analysis runs (spec section 9): torch_ws on GPU + RidgeCV, with --analysis.
#   sbatch -A <account> slurm/bench_round2_analysis.sh [tiers...]   (default: unit-100k unit-100k-raw unit-800k)
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/bench_common.sh
nvidia-smi --query-gpu=name --format=csv,noheader
TIERS=("$@"); [ ${#TIERS[@]} -eq 0 ] && TIERS=(unit-100k unit-100k-raw unit-800k)
for T in "${TIERS[@]}"; do
  for S in torch_ws cpu_ridgecv; do
    echo "=== $S $T"; date
    "$PY" scripts/run_bench.py --solver $S --tier $T --task CV --analysis --tag round2-analysis
  done
done
date
