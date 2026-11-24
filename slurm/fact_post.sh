#!/bin/bash
# factorized-readout: summaries (+ result rows with --write-rows) and figures for a tier, off the login node.
#   sbatch -p gpu_test slurm/fact_post.sh unit-100k [--write-rows]
#SBATCH -J fact-post
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH -t 0-01:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
TIER=$1; shift
$PY -u -m lassobench.factorized.summarize --tier $TIER "$@"
$PY -u scripts/dev_fact_figures.py --tier $TIER
