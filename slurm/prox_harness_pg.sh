#!/bin/bash
# gpu-prox: gradient-optimizer Lasso family through the harness on unit-100k (Task P and CV).
#SBATCH -J prox-hpg
#SBATCH -t 0-10:00
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/%x-%j.out
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
nvidia-smi -L
TIER=${TIER:-unit-100k}
rb() { $PY -u scripts/run_bench.py --tag gpu-prox --tier $TIER --timeout 5000 "$@"; }
for task in P CV; do
rb --task $task --solver torch_prox_adam     --solver-args epochs=300 lr=1e-2
rb --task $task --solver torch_prox_adam     --solver-args epochs=1000 lr=1e-2
rb --task $task --solver torch_prox_sgd      --solver-args epochs=1000 lr_schedule=const
rb --task $task --solver torch_prox_adam_mb  --solver-args epochs=100 lr=1e-2 batch_size=128
rb --task $task --solver torch_subgrad_adam  --solver-args epochs=1000 lr=1e-2 finalize=prox
rb --task $task --solver torch_hadamard_adam --solver-args epochs=1000 lr=1e-2 finalize=prox
done
rb --task P --solver torch_prox_adam --solver-args epochs=300 lr=1e-2 x_on_host=True
rb --task P --solver torch_prox_adam_mb --solver-args epochs=100 lr=1e-2 batch_size=128 x_on_host=True
