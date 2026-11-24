#!/bin/bash
#SBATCH -J bench-cpu
#SBATCH -p sapphire
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o /dev/null
# Account: pass it on the command line (sbatch -A <account> ...). Submit from the repo root: output goes to
# $LASSOBENCH_STORE/results/logs/%x-%j.out (see slurm/bench_common.sh).
# Usage: sbatch -A <account> [-p partition -c N] slurm/bench_cpu.sh "<run_bench.py args>" ["<more args>" ...]
# Each quoted argument is one run_bench.py invocation, run sequentially.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/bench_common.sh
lscpu | grep "Model name"; echo "cpus: $(nproc)"
for a in "$@"; do
  echo "=== run_bench.py $a"; date
  eval "$PY" scripts/run_bench.py $a
done
date
