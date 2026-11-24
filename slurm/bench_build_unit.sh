#!/bin/bash
#SBATCH -J build-unit
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 2:00:00
#SBATCH -o /dev/null
# Account: pass it on the command line (sbatch -A <account> ...). Submit from the repo root: output goes to
# $LASSOBENCH_STORE/results/logs/%x-%j.out (see slurm/bench_common.sh).
# Usage: sbatch -A <account> slurm/bench_build_unit.sh unit-100k unit-800k ...
# Needs write access to LASSOBENCH_DATA and the old repo's loaders (LASSOBENCH_OLD_REPO); see scripts/build_unit_tiers.py.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/bench_common.sh
nvidia-smi --query-gpu=name --format=csv,noheader
"$PY" scripts/build_unit_tiers.py "$@"
