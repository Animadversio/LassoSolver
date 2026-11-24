#!/bin/bash
#SBATCH -J build-synxl
#SBATCH -p serial_requeue
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /dev/null
# Account: pass it on the command line (sbatch -A <account> ...). Submit from the repo root: output goes to
# $LASSOBENCH_STORE/results/logs/%x-%j.out (see slurm/bench_common.sh).
# Usage: sbatch -A <account> slurm/bench_build_synxl.sh     (writes ~41 GB into LASSOBENCH_DATA/syn-xl)
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/bench_common.sh
date
"$PY" scripts/build_tiers.py syn-xl --overwrite
date
