#!/bin/bash
# Holds a GPU node for interactive gpu-prox development (ssh in and run scripts).
#SBATCH -J prox-dev
#SBATCH -t 0-04:00
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
nvidia-smi -L
sleep 4h
