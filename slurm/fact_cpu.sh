#!/bin/bash
# factorized-readout: CPU-side steps (summaries, rows, figures, checks).
#   sbatch slurm/fact_cpu.sh <python args...>     e.g. -m lassobench.factorized.summarize --tier unit-100k
#SBATCH -J fact-cpu
#SBATCH -A binxuwang_lab
#SBATCH -p shared,serial_requeue
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-01:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
/n/home12/binxuwang/.conda/envs/torch2/bin/python -u "$@"
