#!/bin/bash
# gpu-prox scaling table: FISTA over (n, p, T) on one GPU. Submit once per partition:
#   sbatch -A binxuwang_lab -p gpu       slurm/prox_scale_sweep.sh   # A100-SXM4-80GB
#   sbatch -A binxuwang_lab -p gpu_h200  slurm/prox_scale_sweep.sh   # H200
#SBATCH -J prox-sweep
#SBATCH -t 0-08:00
#SBATCH -c 8
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
nvidia-smi -L
F32='["torch_fista",{"tol":1e-4}]'
S5='["torch_fista",{"tol":1e-4,"segments":5}]'
F64='["torch_fista",{"tol":1e-6,"dtype":"float64"}]'
run() { $PY -u scripts/dev_prox_scale.py --tag scale --warmup "$@"; }
run --n 1000  --p 10000   --T 64  --configs "[$F32,$S5,$F64]"
run --n 1000  --p 100000  --T 64  --configs "[$F32,$S5,$F64]"
run --n 1000  --p 100000  --T 384 --configs "[$F32,$S5,$F64]"
run --n 10000 --p 100000  --T 64  --configs "[$F32,$S5,$F64]"
run --n 1000  --p 1000000 --T 64  --configs "[$F32,$S5]"
run --n 10000 --p 1000000 --T 64  --configs "[$F32]"
