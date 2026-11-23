#!/bin/bash
#SBATCH -J bench-cpu
#SBATCH -p sapphire
#SBATCH -A binxuwang_lab
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
# Usage: sbatch [-p partition -c N] slurm/bench_cpu.sh "<run_bench.py args>" ["<more args>" ...]
# Each quoted argument is one run_bench.py invocation, run sequentially.
set -u
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
lscpu | grep "Model name"; echo "cpus: $(nproc)"
for a in "$@"; do
  echo "=== run_bench.py $a"; date
  eval $PY scripts/run_bench.py $a
done
date
