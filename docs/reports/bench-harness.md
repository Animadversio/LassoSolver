# bench-harness report

Session `bench-harness`: benchmark infrastructure, data tiers, reference solutions, CPU baselines, leaderboard.
Every number below was read back from `tables/results.jsonl` (run ids given where useful) or from a job log under
`STORE/results/logs/`. Numbers I could not verify are listed in the last section.

## What exists

| piece | file | notes |
|---|---|---|
| tiers | `lassobench/data.py` (`load_tier`, builders), `scripts/build_tiers.py`, `scripts/build_unit_tiers.py` | memmapped float32 `.npy` under `STORE/data/<tier>/` + `meta.json` |
| runner | `lassobench/runner.py`, CLI `scripts/run_bench.py` | solver discovery, Task P / Task CV, timing, memory, git hash, append-only rows |
| metrics | `lassobench/evaluate.py` | float64 objective / duality gap / suboptimality, test R2, nnz, Jaccard, grid-edge fractions |
| references | `scripts/build_reference.py` -> `STORE/cache/ref/<tier>/` | celer tol 1e-9, then sklearn-CD polish until harness gap <= 1e-8 |
| CPU baselines | `lassobench/solvers/cpu_sklearn_cd.py`, `cpu_celer.py`, `cpu_ridgecv.py`, `cpu_sklearn_legacy.py` | per-target paths over joblib workers |
| report | `lassobench/report.py` -> `tables/leaderboard.md`, `figures/time_vs_subopt_<tier>.png` | `python -m lassobench.report` |
| slurm | `slurm/bench_cpu.sh`, `bench_gpu.sh`, `bench_reference.sh`, `bench_build_unit.sh`, `bench_build_synxl.sh`, `bench_cpu_round2.sh` | logs in `STORE/results/logs/` |
| tests | `tests/test_harness.py` | 6 tests on syn-s (gap formula vs `api.duality_gap`, alpha_max zero solution, P and CV rows); ~105 s on 32 cores |

Harness behaviour worth knowing:

- `load_tier` returns memmaps; the runner pages X into RAM once before timing (load time is reported separately as `time_load_s`).
- Warm-up: one untimed `fit_path` on 4 targets x 5 alphas (`time_warmup_s`); `--warmup none` to skip.
- Prepare time: if a solver has `prepare(X)` it is timed separately; otherwise `info["time_prepare_s"]` is used if the solver reports it.
- Memory: a sampler thread records peak RSS of the process **and its children** (joblib/loky workers) and the process's GPU memory via NVML (plus `torch.cuda.max_memory_allocated`).
- Quality numbers are always recomputed by the harness in float64 from the returned sparse coefficients, never taken from solver self-reports.
- Task CV: `KFold(5, shuffle=True, random_state=0)` on training rows. Default `fit_cv` calls `fit_path` 5 times plus one refit truncated to the largest chosen alpha index. Ties in CV MSE pick the larger alpha.
- `--timeout` records a `status: timeout` row. Errors record `status: error` with the traceback.
- The runner ignores a cached reference whose grid/targets do not match the tier.

## Tiers (on STORE)

| tier | n_train / n_test | p | T | reliable ch. (rel >= 0.5) | source |
|---|---|---|---|---|---|
| `syn-s` | 1000 / 500 | 20,000 | 16 | (all) | `make_synthetic` defaults |
| `pca1k` | 822 / 206 | 1,000 | 64 | 20 | old Xdict pickle, `.layer4.Bottleneck0` pca1000, z-scored |
| `pca1k-raw` | 822 / 206 | 1,000 | 64 | 20 | same, centered only (column std 1.2..28) |
| `unit-100k` | 822 / 206 | 100,352 | 64 | 20 | resnet50_robust `.layer4.Bottleneck0` (2048x7x7), z-scored |
| `unit-100k-raw` | 822 / 206 | 100,352 | 64 | 20 | same X, centered, one global scale |
| `unit-800k` | 822 / 206 | 779,947 | 64 | 20 | `.layer1.Bottleneck0` (256x56x56), 22,869 constant columns dropped, z-scored; **same Y and split as unit-100k** |
| `unit-800k-raw` | 822 / 206 | 779,947 | 64 | 20 | same X, centered, one global scale |
| `unit-800k-383` | 822 / 206 | 779,947 | 383 | 3 | same X, Y = subject `paul_241119-241122` (383 l_STS channels). **Speed-only** |
| `unit-800k-383-raw` | 822 / 206 | 779,947 | 383 | 3 | raw-scaled variant, speed-only |
| `syn-xl` | 10,000 / 1,000 | 1,000,000 | 384 | (all) | chunked `make_synthetic` model, 41 GB |

- Images and their order come from the old pickle. Its responses equal the HDF5 subject `paul_20241119-20241122` exactly (max abs diff 0.0). The split is the old `train_test_split(test_size=0.2, random_state=42)`.
- **Image-order check:** the top 5 principal components of our freshly extracted layer4.B0 features correlate with the pickle's pca1000 columns at |r| = 1.000 (all 5). So the same images were used, in the same order, with the same model.
- The 383-channel subject has the same 1028 stimulus names in the same order, but it is a different array: l_STS, median reliability 0.02 (90th percentile 0.15, max 0.68), only 3 channels >= 0.5. The 64-channel subject is l_cIT: median reliability 0.31, 20 channels >= 0.5. I first built `unit-800k` on the 383-channel subject. On the coordinator's request it was rebuilt on the 64 cIT channels, and the 383 version was renamed `unit-800k-383`. Rows written before the rename (tier `unit-800k*` with T=383) are remapped to `unit-800k-383*` by `report.py`.
- Feature extraction took 85 s on an A100 for both layers (job 63193850). Raw features are cached in `STORE/cache/features/`. Layer1 activations are 57% exact zeros (post-ReLU); layer4 activations 66%.
- `unit-xl` was **not built**: the 5-monkey control-session HDF5 (`vvs-accentuate_controlsessions_5monkeys_250504-250512.h5`, up to 5,595 images x 383 ch for venus) references accentuated images (`AlexNet_training_seed_01_RidgeCV_unit_..._level_..png`) that I could not find under `/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli`. Its reliability values are also near zero (median -0.09..0.02). It needs the image root from whoever generated those stimuli.

## Reference solutions (Task P, float64)

| tier | status | time | max harness gap |
|---|---|---|---|
| `syn-s` | done | 732 s (celer 80 s + sklearn-CD polish) | 5.0e-9 |
| `pca1k` | done | 89 s | 5.0e-9 |
| `unit-100k` | job 63197416 still running (celer pass 1,133 s, max gap 1.2e-7; sklearn-CD polish of all 64 targets at tol 5e-9 has run > 3 h) | - | - |
| `unit-800k` (64 ch) | job 63244685 submitted after the rename | - | - |
| `unit-800k-383` | cancelled after 3 h, still in the celer pass; it would have saved under the old tier name | - | - |
| `syn-xl` | not attempted on CPU | - | - |

Lesson: **celer's own stopping gap is much tighter than the harness gap.** celer uses an extrapolated dual point; the harness rescales the residual. At celer tol 1e-9, some (alpha, target) problems still have a harness gap of 6.9e-6 on pca1k. The CD polish fixes this, but it is slow at p = 1e5 and above. The first reference attempt also stalled at gap 5e-7 because the solvers returned float32 coefficients; they now return float64.

## CPU baselines

Hardware: serial_requeue nodes, either Xeon Platinum 8268 (2.9 GHz) or 8358 (2.6 GHz), 32 cores, as recorded per row. A few dev-node runs shared the node with another run, noted where relevant.

### Unit basis, Task CV at tol 1e-4 (headline: test R2 over the 20 reliable cIT channels)

| tier | solver | hardware | time (s) | R2 reliable | R2 all 64 | nnz median | run |
|---|---|---|---|---|---|---|---|
| unit-100k | cpu_celer | 8358 x32 | 403 | 0.3257 | - | 120.5 | 18278e |
| unit-100k | torch_ws (sibling, for comparison) | H200 | 33.2 | 0.3257 | - | 120.5 | 380482 |
| unit-100k | cpu_ridgecv | 8268 x32 | 2.8 | **0.3468** | 0.1642 | dense | 42aace |
| unit-100k-raw | cpu_celer (16 jobs) | 8268 x16 | 948 | 0.3159 | 0.1494 | 105 | d78efd |
| unit-100k-raw | cpu_ridgecv | 8268 x32 | 3.7 | **0.3515** | 0.1668 | dense | a95b5f |
| unit-800k | cpu_ridgecv | 8358 x32 | 10.3 | 0.1253 | 0.0572 | dense | b2a1b9 |
| unit-800k-raw | cpu_ridgecv | 8358 x32 | 8.8 | 0.0757 | 0.0328 | dense | 7c1702 |
| unit-800k | cpu_celer / cpu_sklearn_cd | 8358 x32 | running (4 h cap) | | | | jobs 63239467 / 63239490 |
| unit-800k-raw | cpu_celer | 8358 x32 | running (4 h cap) | | | | job 63239509 |
| unit-100k | cpu_sklearn_cd | 8358 x32 | running (4 h cap) | | | | job 63237091 |
| unit-800k-383 | cpu_celer / cpu_sklearn_cd / raw celer | 8358 x32 | running (4 h cap) | | | | jobs 63237102 / 63237105 / 63237109 |

- celer and torch_ws pick the same alphas and reach the same test R2 to 4 decimals on unit-100k. This cross-checks both the GPU solver and the harness CV. CPU celer is 12x slower than torch_ws on an H200 there.
- **Lasso vs Ridge in the unit basis: Ridge wins on layer4.** Reliable-channel R2 is 0.347 for Ridge vs 0.326 for Lasso on z-scored unit-100k, and 0.352 vs 0.316 on raw. Ridge also fits in 3 s.
- Layer1 (unit-800k) predicts the cIT channels much worse than layer4 (Ridge 0.125 vs 0.347), as expected for an early layer.

### Raw vs z-scored (spec v0.1, section 7)

| features | Lasso R2 reliable (z / raw) | Ridge R2 reliable (z / raw) |
|---|---|---|
| PCA-1000 (pca1k) | 0.258 / 0.309 | 0.003 / 0.352 |
| unit-100k (layer4) | 0.326 / 0.316 | 0.347 / 0.352 |
| unit-800k (layer1) | pending | 0.125 / 0.076 |

- On **PCA scores**, z-scoring is harmful. It inflates the ~900 low-variance (mostly noise) components to the same scale as the top components. Ridge, being isotropic, then cannot shrink them selectively and falls to R2 ~0: 25% of channels choose the largest ridge alpha, i.e. predict 0. Widening the ridge grid from 1e-6..1e4 to 1e-4..1e8 (relative) changed nothing. This is a property of z-scored PCA features, not a bug.
- In the **unit basis** the effect is small and has no consistent sign at layer4 (Lasso +0.010 for z, Ridge +0.005 for raw). At layer1, z-scoring helps Ridge (0.125 vs 0.076). Raw unit features have very unequal column scales (post-ReLU, many near-dead units), so z-scoring up-weights rare-firing units there.
- Sanity check of the raw-tier construction: Ridge is rotation invariant. Ridge on `unit-100k-raw` (0.3515) equals Ridge on `pca1k-raw` (0.3518) to 3 decimals, as it should: the PCA scores are a rotation of the same centered layer4 features.
- **Recommendation:** keep z-scoring as the default for the unit-basis tiers (it is what the solvers are tuned for), and keep the `-raw` tiers as a required secondary report. Final call when the unit-800k Lasso rows land.

### Standardization and speed (pca1k, Platinum 8268 x32)

| | z-scored (`pca1k`) | raw (`pca1k-raw`) |
|---|---|---|
| sklearn CD, Task P tol 1e-4 | 4.7 s | 2.7 s |
| sklearn CD, Task P tol 1e-6 | 15.1 s | 5.2 s |
| celer, Task P tol 1e-4 | 13.0 s | 9.5 s |
| sklearn CD, Task CV tol 1e-4 | 29.1 s | 20.8 s |
| celer, Task CV tol 1e-4 | 39.3 s | 25.2 s |
| CV-chosen nnz (median) | 34 | 48 |
| Lasso test R2 all / reliable | 0.114 / 0.258 | 0.147 / 0.309 |

So "X was not standardized" was **not** the cause of the old slow fits. On PCA features the raw scaling is both faster (Lasso solutions are sparser along the path: Task P median nnz 52 vs 472) and more predictive.

### Old-protocol reproduction (pca1000)

`cpu_sklearn_lassocv_legacy` reruns the old `MultiOutputSeparateLassoCV(cv=5, n_alphas=100, n_jobs=16)`: per-target LassoCV with its own grid and unshuffled folds, sequential over targets.

| tier | max_iter | time | R2 all / reliable |
|---|---|---|---|
| pca1k-raw (the old setting) | 10000 | 651 s | 0.1469 / 0.306 |
| pca1k-raw | 1000 (old default) | 580 s | 0.1469 / 0.306 |
| pca1k | 10000 | 443 s | 0.1159 / 0.263 |
| pca1k | 1000 | 529 s | 0.1159 / 0.263 |

- The old timing of 112-223 s is **not reproduced**: ours is 2.6-5.8x slower on 16 cores of a Platinum 8268. Possible reasons: different hardware, n_jobs=-1 meaning more cores then, and the dev node being shared with another run during the max_iter=1000 reruns (which explains 529 vs 443 s). The predictive result does match the old notebooks.
- The harness's 50-alpha, shuffled-5-fold protocol gives nearly the same R2 (0.309 vs 0.306 reliable on raw) in 21 s (sklearn CD) instead of 580-651 s. Most of the old cost was the protocol: 100 alphas x 5 folds x 64 targets, sequential over targets, and 16-way parallelism only over folds.

## Things about the data that surprised me

- **SNR is low and very uneven.** 20/64 cIT channels have reliability >= 0.5. The all-channel mean R2 (0.15-0.17) is about half the reliable-channel mean (0.31-0.35). On the 383-channel l_STS array almost nothing is predictable (all-channel R2 0.01-0.02).
- **CV-optimal Lasso solutions are very sparse:** median 34 (pca1k) to 120 (unit-100k) nonzeros out of 1e3-1e5 features. A sizeable minority of channels choose alpha_max, i.e. the empty model: 19% on unit-100k, 16-19% on pca1k, 28% on the 383-ch tier. **Every grid-edge hit is at the largest alpha, never at the smallest** (column "alpha at grid edge: smallest / largest" in the leaderboard), so the spec's eps = 1e-2 floor is not binding. Those channels are unpredictable, and the predict-zero choice is correct for them.
- **Ridge beats Lasso in R2 on these data.** Sparsity costs about 0.02-0.04 R2 on the reliable channels at layer4. Lasso's value here is interpretability and compactness (120 vs 100,352 weights), not accuracy.
- Lasso paths in the unit basis are expensive at the small-alpha end: Task P median nnz is ~510 on unit-100k at alpha_min, near n = 822.

## Bugs found and fixed in my code

1. **`cpu_ridgecv` gave test R2 -5.9 / -7.8 on pca1k / pca1k-raw.** X is centered on the training rows, which puts the all-ones vector in the null space of XXᵀ. With `fit_intercept=False`, sklearn's LOO-GCV error goes to 0 as alpha → 0 along that direction, so every target picked the smallest alpha (100% of channels at the grid floor). Fixed with `fit_intercept=True` (the fitted intercept is < 3e-9, ignored at prediction). `pca1k-raw` now gives 0.352 reliable (passes the spec v0.1 sanity check; old notebooks: 0.164 all-channel, ours 0.167). Pre-fix ridge rows are dropped by `report.py` (`SUPERSEDED`).
2. Solvers cast coefficients to float32, which capped the achievable harness gap near 5e-7. They now return float64.
3. celer under joblib threads does not parallelize (it holds the GIL): 124 s vs 15 s with processes on pca1k. It now runs in loky processes that share one copy-on-write memmap of X in `/dev/shm`. Per-worker copies would need 32 x 5 GB at p = 780k.

## Contract / spec change requests

1. **CV protocol in `api.py`.** The runner already implements an optional `fit_cv(X, Y, alphas, folds) -> dict(coef, best_idx, cv_mse, info)`, with `folds` = the harness list of (train_idx, val_idx). Please add it to `Solver` so siblings can rely on it.
2. **Convergence criterion.** The harness gap uses the plain rescaled-residual dual point, which is pessimistic. celer at tol 1e-9 still shows harness gaps of ~1e-5 on some low-alpha problems while being optimal to 1e-8 by suboptimality. Suggest either (a) defining "converged" by suboptimality vs the reference when one exists (the leaderboard already reports both), or (b) taking the dual point as the best of the rescaled residual and the solver's returned dual (`info["theta"]`) after the harness checks feasibility.
3. **sklearn's tol semantics.** sklearn stops at gap <= tol * ||y||², which equals harness gap_rel <= 2*tol. The baseline passes tol/2, so its rows sit right at gap_rel ~ 0.99999 x tol. Ranking at "<= 1e-4" is fine, but a solver that stops at exactly tol can flip on float rounding. Suggest ranking at gap_rel <= 1.05 x tol.
4. **Reliability threshold.** 0.5 leaves 20/64 cIT and 3/383 STS channels. For any 383-channel tier the headline should be a lower threshold (e.g. 0.3: 13 channels) or the ncsnr-weighted R2. Otherwise keep 383-channel tiers speed-only, as now.
5. **unit-xl** needs the image root of the control-session stimuli (see Tiers).

## Numbers not verified / open

- The unit-800k (64 ch) and unit-800k-383 Lasso CV rows, and sklearn CD on unit-100k, were still running when this was written. The leaderboard picks them up on regeneration.
- The unit-100k and unit-800k references have not finished, so subopt columns are empty for the unit tiers.
- The old 112-223 s LassoCV timing and the old MultiTaskLassoCV 12 s were not reproduced (the latter was not attempted).
- syn-xl has no CPU baseline and no reference. A CPU run at p = 1e6, T = 384 is far beyond the 4 h cap (see the unit-800k-383 celer CV row: 4 h cap).
