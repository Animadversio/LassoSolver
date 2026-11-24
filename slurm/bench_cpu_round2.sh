#!/bin/bash
# Round-2 CPU baselines (bench-harness). Submits one job per heavy run on serial_requeue (32 cores, 4 h cap per run).
cd /n/home12/binxuwang/Github/LassoSolver
S=slurm/bench_cpu.sh
SUB="sbatch -p serial_requeue -c 32 --mem=180G -t 20:00:00 --requeue"
T=14400
$SUB -J r2-u100k $S \
  "--solver cpu_celer --tier unit-100k --task CV --solver-args tol=1e-4 --timeout $T --tag round2" \
  "--solver cpu_sklearn_cd --tier unit-100k --task CV --solver-args tol=1e-4 --timeout $T --tag round2" \
  "--solver cpu_ridgecv --tier unit-100k --task CV --tag round2" \
  "--solver cpu_ridgecv --tier unit-100k-raw --task CV --tag round2" \
  "--solver cpu_celer --tier unit-100k-raw --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
# unit-800k = 64 l_cIT channels (statistical tier); unit-800k-383 = speed-only (renamed 2026-11-23; jobs submitted
# before the rename ran on the 383-ch data under the old name and are remapped by report.py)
$SUB -J r3-u800k-celer $S "--solver cpu_ridgecv --tier unit-800k --task CV --tag round2" \
  "--solver cpu_celer --tier unit-800k --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
$SUB -J r3-u800k-skl $S "--solver cpu_sklearn_cd --tier unit-800k --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
$SUB -J r3-u800k-raw $S "--solver cpu_ridgecv --tier unit-800k-raw --task CV --tag round2" \
  "--solver cpu_celer --tier unit-800k-raw --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
