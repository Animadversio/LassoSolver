#!/bin/bash
#SBATCH -J bench-dev
#SBATCH -p serial_requeue
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 12:00:00
#SBATCH -o /dev/null
# Account: pass it on the command line (sbatch -A <account> ...). Submit from the repo root: output goes to
# $LASSOBENCH_STORE/results/logs/%x-%j.out (see slurm/bench_common.sh).
# Placeholder allocation for interactive dev over ssh: sbatch -A <account> slurm/bench_devnode.sh, then
# ssh to the node from squeue and run scripts there (export PYTHONNOUSERSITE=1).
sleep infinity
