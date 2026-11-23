#!/bin/bash
#SBATCH -J bench-gpu
#SBATCH -p kempner_requeue
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 6:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
# Usage: sbatch slurm/bench_gpu.sh "<run_bench.py args>" ...   (same as bench_cpu.sh, on one GPU)
set -u
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for a in "$@"; do
  echo "=== run_bench.py $a"; date
  eval $PY scripts/run_bench.py $a
done
date
