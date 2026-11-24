# Lasso benchmark spec (v0.1)

Owned by the coordinator session. Sub-sessions: propose changes in your report, do not edit.

## 1. Problem

For every target (channel) t and every alpha on its grid, independently:

    min_w  1/(2n) ||y_t - X w||^2 + alpha ||w||_1          (sklearn convention, no intercept)

- X (n, p) float32: DNN unit activations, **unit basis**, column z-scored on the training rows,
  constant columns dropped. Y (n, T) float32: per-channel centered (and z-scored) responses.
- Regime: n = 1e3..1e4, p = 1e3..1e6+, T = 64..384. p >> n, highly correlated columns, low SNR.
- Each channel gets its own alpha via K-fold CV. This means the real workload is
  `T targets x A alphas x (K folds + 1 refit)` Lasso problems sharing one X: about 20k-100k problems.
  Exploiting that sharing (batching over targets/alphas/folds, shared Gram/factorizations, warm starts,
  screening) is the whole game.

## 2. Tasks

**Task P (path accuracy).** Given X, Y and the fixed grid `alpha_grid(X, Y, n_alphas=50, eps=1e-2)`,
return all A x T solutions. Scored on wall-clock and on solution quality vs a reference solution
(celer or sklearn CD at tol 1e-8 in float64, cached on STORE).

**Task CV (end to end, the headline number).** Given train X, Y: 5-fold CV over the same grid, pick alpha
per target by mean held-out MSE, refit on all training rows at the chosen alpha, predict the held-out test set.
Folds are fixed by the harness (seeded) so every solver sees the same splits. The default implementation
calls `fit_path` K+1 times; a solver may override `fit_cv` to batch folds, as long as the result is the same.

## 3. Metrics (one row per solver x dataset x hardware in `tables/results.jsonl`)

- `time_total_s`: wall-clock including host->GPU transfer and all precomputation. Also `time_prepare_s`
  (target-independent precompute, reusable across Y) and `time_solve_s` separately. Warm-up/compile runs
  excluded but reported. Peak host RAM and peak GPU memory.
- Optimization quality (Task P): relative objective suboptimality `(P - P_ref)/P_ref` and duality gap
  relative to `||y_t||^2/(2n)`; report median and max over (alpha, target). A solver "converged" if
  max relative gap <= 1e-4 (loose) or 1e-6 (tight). Report time-to-loose and time-to-tight.
- Statistical quality (Task CV): test R2 per channel (1 - SSE/SST, as in the old repo's
  `compute_D2_per_unit`), mean and median over channels, and over reliable channels; selected alpha;
  nnz per channel; support Jaccard vs reference at the selected alpha.
- Hardware: cpu model + thread count, or GPU model. Solver config (tol, max_iter, dtype, extras). Git hash.

Approximate solvers (e.g. gradient/Adam-type) are allowed and interesting: they do not need to hit tight
gaps, but must report where they land on the (time, test R2, nnz) trade-off against exact solvers.
"Sparse" means exact zeros; a method that needs post-hoc thresholding must say so and count that step.

## 4. Dataset tiers

Stored under `STORE/data/<tier>/` as `X_train.npy, Y_train.npy, X_test.npy, Y_test.npy, meta.json`
(float32, memory-mappable). Built by the harness session.

| tier | n_train | p | T | source |
|---|---|---|---|---|
| `syn-s` | 1e3 | 2e4 | 16 | `make_synthetic`, for CI and dev |
| `pca1k` | 822 | 1000 | 64 | old dev-notebook Xdict pkl (paul_20241119-20241122, resnet50_robust layer4.B0 pca1000); reproduces old timings (sklearn per-target LassoCV 112 s, MultiTaskLassoCV 12 s, RidgeCV test R2 ~0.16) |
| `unit-100k` | ~800 | ~1e5 | 64 | same subject/model, layer4 full unit basis (2048x7x7) |
| `unit-800k` | ~800-900 | ~8e5 | 64 or 383 | resnet50_robust layer1.B0 full (256x56x56) |
| `unit-xl` | as large as available (control sessions up to 5575 imgs) | >= 1e6 (concat layers) | 383 | stretch tier |
| `syn-xl` | 1e4 | 1e6 | 384 | synthetic at target scale (X = 40 GB float32), stress test |

Always also fit **RidgeCV** on each real tier as the predictive baseline Lasso has to justify itself against.

## 5. Solver contract

See `lassobench/api.py`: subclass `Solver`, implement `fit_path(X, Y, alphas) -> PathResult` with
`coefs[a]` a scipy CSC (p, T). One file per solver under `lassobench/solvers/`, registered by name in a
dict the harness discovers (`SOLVERS = {"name": Class}` at module level). Helpers `alpha_grid`,
`objective`, `duality_gap`, `make_synthetic` are in `api.py`.

## 6. Prior results to beat (Dec 2024, 16 CPU threads, pca1000 only, 64 targets, 100 alphas, 5 folds)

- sklearn per-target LassoCV: 112-223 s. celer: 179 s (no multithread gain). cuML Lasso via GridSearchCV: ~75x
  slower than CPU (500 tiny GPU fits through joblib; never batched).
- Full unit basis (1e5-8e5 features) was never attempted with Lasso. That is the gap this repo fills.
- X was not standardized then (PCA column std 1..48), a likely cause of the slow/non-converging fits.

## 7. Changes in v0.1 (2026-11-23, after the first round)

- **Headline statistical metric is test R2 over reliable channels** (`r2_test_mean_reliable`, reliability >= 0.5).
  The all-channel mean is dominated by unreliable channels (unit-800k: 0.015 over all 383 channels vs 0.405 over
  reliable ones) and is reported only as a secondary number.
- **Feature scaling is an open question, not a given.** On PCA features z-scoring lowered test R2 (0.147 -> 0.114).
  Every real tier therefore gets a `-raw` variant (columns centered, one global scale, no per-column z-score), and
  solvers must not assume unit column variance: use per-column norms where the algorithm needs them (CD step sizes,
  screening rules). The contract in `api.py` is otherwise unchanged. Which scaling becomes the default is decided
  from the unit-basis comparison.
- **A baseline has to pass a sanity check before it counts.** RidgeCV must give positive test R2 on reliable
  channels on `pca1k-raw` (old notebooks: about 0.16 over all channels); a negative value is a harness bug.
- **Leaderboard rows need committed solver code.** A row counts only if the solver file is committed at the row's
  `git_hash` (not listed in `git_dirty`). Dev runs are welcome in `results.jsonl` but are not ranked.
- **Like-for-like comparisons.** Rank at a fixed tolerance (relative gap 1e-4) on the same hardware, and always
  include at least one CPU reference (sklearn CD or celer) on the same tier, or its recorded timeout.
