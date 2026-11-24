#!/bin/bash
# Run a list of dev_ws_scale.py configs sequentially (one per line of $1) on this node; log to $2.
cd /n/home12/binxuwang/Github/LassoSolver
export PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=16
while read -r line; do
  [ -z "$line" ] && continue
  echo "### $line" >> "$2"
  timeout 5400 /n/home12/binxuwang/.conda/envs/torch2/bin/python scripts/dev_ws_scale.py $line >> "$2" 2>&1
done < "$1"
