# LassoSolver

A benchmark for large-scale Lasso solvers on neural-prediction regression, aimed at finding the fastest one on
CPU, A100, H100 or H200 GPUs.
Y holds neuronal responses (n images x T channels), X holds DNN unit activations in the unit basis
(p = 1e3 to 1e6 features, p >> n). Each channel gets its own Lasso path and its own alpha chosen by 5-fold CV.
The problem, metrics and solver contract are in [`docs/BENCHMARK_SPEC.md`](docs/BENCHMARK_SPEC.md) and
[`lassobench/api.py`](lassobench/api.py). Results so far are in [`tables/leaderboard.md`](tables/leaderboard.md),
and the harness/baseline findings in [`docs/reports/bench-harness.md`](docs/reports/bench-harness.md).

## Setup

```bash
git clone https://github.com/Animadversio/LassoSolver && cd LassoSolver
pip install -r requirements.txt        # or use an env that already has these (see the file for versions)
```

Three environment variables decide where things live:

| variable | meaning | default |
|---|---|---|
| `LASSOBENCH_DATA` | where tiers are **read** from (can be read-only) | `$LASSOBENCH_STORE/data` |
| `LASSOBENCH_STORE` | **writable** dir for raw result dumps, reference cache, logs | the original author's lab dir |
| `LASSOBENCH_RESULTS` | the JSON-lines result log your runs append to | `tables/results.jsonl` in the repo |

On the Kempner cluster, members of `kempner_alvarez_lab` can use the shared read-only copy of the tiers:

```bash
export LASSOBENCH_DATA=/n/holylabs/LABS/kempner_alvarez_lab/Lab/shared/LassoSolver/data
export LASSOBENCH_STORE=/path/you/can/write/LassoSolver        # e.g. under your lab or scratch space
export LASSOBENCH_RESULTS=$PWD/tables/results_$USER.jsonl      # your own log, no merge conflicts
export LASSOBENCH_PY=$(which python)                           # used by the slurm scripts
```

Cached reference solutions (for the suboptimality metrics) are read from `$LASSOBENCH_STORE/cache/ref`, then from
`$LASSOBENCH_DATA/../cache/ref`.

## Tiers

| tier | n_train | p | T | what |
|---|---|---|---|---|
| `syn-s` | 1000 | 20,000 | 16 | synthetic, for tests |
| `pca1k`, `pca1k-raw` | 822 | 1,000 | 64 | old PCA-1000 features (regression check only) |
| `unit-100k`, `-raw` | 822 | 100,352 | 64 | resnet50_robust layer4.B0 units, 20 reliable channels |
| `unit-800k`, `-raw` | 822 | 779,947 | 64 | resnet50_robust layer1.B0 units, same channels |
| `unit-800k-383`, `-raw` | 822 | 779,947 | 383 | same X, 383 l_STS channels (speed only, 3 reliable) |
| `syn-xl` | 10,000 | 1,000,000 | 384 | synthetic at target scale (41 GB) |

`-raw` = columns centered with one global scale instead of per-column z-scoring. List what is available with
`python scripts/run_bench.py --list`.

## Run one solver on one tier

```bash
# quick CPU check (a few channels, seconds)
python scripts/run_bench.py --solver cpu_sklearn_cd --tier pca1k --task P --max-targets 4
# Task CV: 5-fold CV over the 50-alpha grid, refit, test R2 (the headline metric)
python scripts/run_bench.py --solver cpu_ridgecv --tier unit-100k --task CV
python scripts/run_bench.py --solver cpu_celer   --tier unit-100k --task CV --solver-args tol=1e-4 n_jobs=32
# add --analysis to a CV run for the sparsity frontier, support stability, 1-SE rule and relaxed refit
```

Each run appends one row to `$LASSOBENCH_RESULTS` (time, memory, hardware, git hash, metrics). `--max-targets`
runs are dev runs and are not ranked. Solvers are discovered from the `SOLVERS = {...}` dict in each
`lassobench/solvers/*.py`; add yours the same way.

On Slurm (submit from the repo root; output goes to `$LASSOBENCH_STORE/results/logs/<job>-<id>.out`):

```bash
sbatch -A <account> -p <cpu partition> slurm/bench_cpu.sh "--solver cpu_celer --tier unit-100k --task CV"
sbatch -A <account> -p <gpu partition> slurm/bench_gpu.sh "--solver torch_ws --tier unit-100k --task CV"
```

Each quoted string is one `run_bench.py` call, and several strings run in sequence.

## Leaderboard and figures

```bash
python -m lassobench.report                                   # from $LASSOBENCH_RESULTS or tables/results.jsonl
python -m lassobench.report tables/results.jsonl tables/results_$USER.jsonl --out my_leaderboard.md --figdir my_figs
```

## Tests

```bash
python -m pytest tests -q        # needs the syn-s tier; ~2 min on 32 cores
```
