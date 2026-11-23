# LassoSolver

Goal: the fastest large-scale Lasso solver for neural-prediction regressions, on whatever hardware
wins (CPU, A100, H100, H200). Read `docs/BENCHMARK_SPEC.md` before writing any solver or benchmark code;
it defines the problem, the solver contract (`lassobench/api.py`), the metrics and the result format.

## Use case
- Y: noisy neuronal responses, n images x T channels (T = 64 to 384), n ~ 1e3 to 1e4.
- X: DNN features in the **unit basis** (no PCA/SRP), p = 1e3 to 1e6+ features. p >> n.
- Want weights sparse per unit, one independent Lasso per channel, each with its own alpha chosen by CV.
- Later direction (not now): factorized readout, spatially sparse x feature-wise dense.

## Where things go
- Code, figures, tables, short summaries: this repo (home dir, small quota). Never write matrices here.
- Big data, caches, raw result dumps: `/n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver/{data,cache,results}`
  (exported as `lassobench.api.STORE`).
- Prior work and data loaders: `/n/home12/binxuwang/Github/Closed-loop-visual-insilico` (read-only for us).

## Compute
- Python: `/n/home12/binxuwang/.conda/envs/torch2/bin/python` (torch 2.4.1, celer 0.7.3, cuml 24.10, cupy 13.3,
  sklearn 1.6, numba, jax). skglm / glmnet are not installed; `pip install` into this env is fine for pure-python solver libs.
- Login node: only quick tests (< ~1 min, small). Importing cuml/cudf on the login node is very slow and has no GPU.
- Everything else: Slurm. Accounts `binxuwang_lab` (better fairshare) and `konkle_lab`. GPU partitions include
  `kempner`, `kempner_h100`, `kempner_h200`, `kempner_requeue`, `gpu`, `gpu_h200`, `gpu_requeue`, `gpu_test`;
  check with `sbatch --test-only` which start soonest. Record the GPU model with every result.
- Keep sbatch scripts in `slurm/`, logs under `STORE/results/logs/`.

## Multi-session rules
Several Claude sessions share this checkout, coordinated by a main session.
- Each session owns specific files (named in its brief). Do not edit files you do not own; `lassobench/api.py`
  and `docs/BENCHMARK_SPEC.md` belong to the coordinator: propose changes in your report instead.
- Commit only your own files (`git add <explicit paths>`), never `git add -A`. Do not switch branches.
- Report results as rows in `tables/results.jsonl` via the harness (append-only) plus a short
  `docs/reports/<session-name>.md` with what was tried, what worked, what failed, and timing numbers.
