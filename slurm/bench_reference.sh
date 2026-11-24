#!/bin/bash
#SBATCH -J bench-ref
#SBATCH -p sapphire
#SBATCH -c 32
#SBATCH --mem=240G
#SBATCH -t 2-00:00:00
#SBATCH -o /dev/null
# Account: pass it on the command line (sbatch -A <account> ...). Submit from the repo root: output goes to
# $LASSOBENCH_STORE/results/logs/%x-%j.out (see slurm/bench_common.sh).
# Usage: sbatch -A <account> slurm/bench_reference.sh --tier unit-100k [build_reference.py args]
# Saves to $LASSOBENCH_STORE/cache/ref/<tier>/ (a provisional celer-stage reference first, then the polished one).
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/bench_common.sh
lscpu | grep "Model name"; nproc; date
"$PY" scripts/build_reference.py "$@"
date
