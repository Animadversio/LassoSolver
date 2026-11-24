#!/bin/bash
# Round-2 CPU baselines (bench-harness): submits one job per heavy run (32 cores, 4 h cap per run).
#   ACCOUNT=<account> bash slurm/bench_cpu_round2.sh       (run from the repo root)
# Jobs submitted before the 2026-11-23 rename ran `unit-800k` on the 383-ch data; report.py remaps those rows.
set -u
cd "$(dirname "$0")/.."
S=slurm/bench_cpu.sh
SUB="sbatch -A ${ACCOUNT:?set ACCOUNT} -p ${PARTITION:-serial_requeue} -c 32 --mem=180G -t 20:00:00 --requeue"
T=14400
$SUB -J r2-u100k $S \
  "--solver cpu_celer --tier unit-100k --task CV --solver-args tol=1e-4 --timeout $T --tag round2" \
  "--solver cpu_sklearn_cd --tier unit-100k --task CV --solver-args tol=1e-4 --timeout $T --tag round2" \
  "--solver cpu_ridgecv --tier unit-100k --task CV --tag round2" \
  "--solver cpu_ridgecv --tier unit-100k-raw --task CV --tag round2" \
  "--solver cpu_celer --tier unit-100k-raw --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
$SUB -J r3-u800k-celer $S "--solver cpu_ridgecv --tier unit-800k --task CV --tag round2" \
  "--solver cpu_celer --tier unit-800k --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
$SUB -J r3-u800k-skl $S "--solver cpu_sklearn_cd --tier unit-800k --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
$SUB -J r3-u800k-raw $S "--solver cpu_ridgecv --tier unit-800k-raw --task CV --tag round2" \
  "--solver cpu_celer --tier unit-800k-raw --task CV --solver-args tol=1e-4 --timeout $T --tag round2"
$SUB -J r2-u800k383 $S "--solver cpu_celer --tier unit-800k-383 --task CV --solver-args tol=1e-4 --timeout $T --tag round2" \
  "--solver cpu_ridgecv --tier unit-800k-383 --task CV --tag round2"
