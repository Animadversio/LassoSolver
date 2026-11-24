#!/bin/bash
# usage: scripts/dev_prox_ssh.sh NODE <python args...>   (runs the torch2 python in the repo on NODE)
NODE=$1; shift
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $NODE \
  "cd /n/home12/binxuwang/Github/LassoSolver && PYTHONNOUSERSITE=1 /n/home12/binxuwang/.conda/envs/torch2/bin/python -u $(printf '%q ' "$@")" 2>&1 | grep -v 'module: command'
