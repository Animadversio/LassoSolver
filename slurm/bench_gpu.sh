#!/bin/bash
#SBATCH -J bench-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 6:00:00
#SBATCH -o /dev/null
# Account: pass it on the command line (sbatch -A <account> ...). Submit from the repo root: output goes to
# $LASSOBENCH_STORE/results/logs/%x-%j.out (see slurm/bench_common.sh).
# Usage: sbatch -A <account> [-p gpu_h200] slurm/bench_gpu.sh "<run_bench.py args>" ...   (one GPU)
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/bench_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for a in "$@"; do
  echo "=== run_bench.py $a"; date
  eval "$PY" scripts/run_bench.py $a
done
date
