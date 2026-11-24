#!/bin/bash
# usage: scripts/dev_ws_run.sh NODE script.py args...   (runs on a held dev GPU node)
NODE=$1; shift
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$NODE" "cd /n/home12/binxuwang/Github/LassoSolver && PYTHONNOUSERSITE=1 V=$V KW='$KW' PYTHONPATH=. OMP_NUM_THREADS=16 /n/home12/binxuwang/.conda/envs/torch2/bin/python $*" 2>&1 | grep -v 'module: command'
