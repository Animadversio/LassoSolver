#!/bin/bash
# round-2 post-processing: faithful l3 (stability figure), clipped l3 (splits figure + rows), unit-100k refresh
PY=/n/home12/binxuwang/.conda/envs/torch2/bin/python
set -e
$PY -m lassobench.factorized.summarize_round2 --tier unit-200k-l3
$PY -m lassobench.factorized.summarize_round2 --tier unit-200k-l3 --dir round2_clip --write-rows --tag factorized-readout-round2-clip
$PY -m lassobench.factorized.summarize_round2 --tier unit-100k
