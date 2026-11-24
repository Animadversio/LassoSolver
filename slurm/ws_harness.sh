#!/bin/bash
#SBATCH -J ws-harness
#SBATCH -A binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/results/logs/ws-harness-%j.out
# Run the harness (scripts/run_bench.py) for each line of the file $1 (args after run_bench.py).
# Rows go to tables/results.jsonl unless the line has --out.
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=16
nvidia-smi --query-gpu=name,memory.total --format=csv
while read -r line; do
  [ -z "$line" ] && continue
  echo "### $line"
  /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/run_bench.py $line
done < "$1"
