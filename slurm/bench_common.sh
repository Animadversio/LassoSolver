# Sourced by slurm/bench_*.sh after `cd` to the repo root. Sets PY and STORE and tees all output to
# $STORE/results/logs/<job name>-<job id>.out.
#   LASSOBENCH_PY     python to use (default: python on PATH)
#   LASSOBENCH_STORE  writable store for results, raw dumps, caches, logs (default: lassobench.api.STORE)
#   LASSOBENCH_DATA   where tiers are read from (default: $LASSOBENCH_STORE/data)
#   LASSOBENCH_RESULTS result log (default: tables/results.jsonl in the repo)
export PYTHONNOUSERSITE=1   # ~/.local packages can shadow the env's numpy on compute nodes
PY=${LASSOBENCH_PY:-python}
STORE=$("$PY" -c 'from lassobench.api import STORE; print(STORE)') || { echo "cannot import lassobench with $PY"; exit 1; }
LOGDIR="$STORE/results/logs"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/${SLURM_JOB_NAME:-bench}-${SLURM_JOB_ID:-local}.out") 2>&1
echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  repo $(pwd)  python $PY  store $STORE  data ${LASSOBENCH_DATA:-$STORE/data}"
