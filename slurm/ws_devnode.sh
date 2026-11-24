#!/bin/bash
#SBATCH -J ws-dev
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/ws-dev-%j.out
# Holds a GPU node for interactive development of the gpu-workingset solvers (ssh in, run scripts).
nvidia-smi
sleep 43000
