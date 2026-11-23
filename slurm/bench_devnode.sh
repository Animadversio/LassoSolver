#!/bin/bash
#SBATCH -J bench-dev
#SBATCH -p serial_requeue
#SBATCH -A binxuwang_lab
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
# Placeholder allocation for interactive dev over ssh (bench-harness session).
sleep infinity
