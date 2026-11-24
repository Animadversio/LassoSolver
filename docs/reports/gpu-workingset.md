# gpu-workingset report (rounds 1 and 2)

Session `gpu-workingset`. Files: `lassobench/solvers/torch_ws.py`, `torch_admm.py`, `_ws_common.py`,
`scripts/dev_ws_*`, `slurm/ws_*.sh`. Every number below was read from run output: harness rows in
`tables/results.jsonl` (tags `gpu-workingset*`, `ws-r2-*`), dev rows in `STORE/results/ws_dev/*.jsonl`,
logs in `STORE/results/logs/` (`grid_*.log`, `cvstop_*.log`, `variants_*.log`, `harness*_*.log`).
Commits: 08f7127 (round 1), b9f84c1 / 99f72ff / e73ac90 (round 2).

**Bottom line.**
- `torch_ws` solves the full 50-alpha path to a verified gap of 1e-4 in 14–27 s (H200) for n = 1e3,
  T = 384 and p = 1e4 to 1e6. The time is almost flat in p: the full X^T R pass is ~1 s of it.
- celer on 32 CPU cores projects to 511 s at p = 1e4 and 1310 s at p = 1e5.
- Task CV on unit-100k went from 34 s to 11.5 s through CV early stopping, which reproduces the
  full-path selection exactly. Fold batching fixed a 5x A100 regression.
- For the sparsity goal, the relaxed Lasso with a CV-chosen ridge penalty moves the R2-vs-k frontier up at
  small k (unit-100k k=10: 0.125 → 0.221, k=30: 0.220 → 0.282) for about 3.4 s extra.
- Drop ADMM.

## 1. What was built

**`torch_ws`**: a celer-style working-set solver, batched over all (target, fold) problems sharing one X on the GPU.

Per alpha, for all active problems at once:

1. *Full check*, the only O(n p B) step: R = Y - XW, then C = R^T X in feature chunks (fp32, TF32 off).
   Each chunk yields three things; (p, B) is never stored:
   - max_j |c_j|, which gives the residual-rescaled dual point and so the duality gap;
   - the top-M working-set candidates (celer score (|c_j| - lam)/||x_j||, valid for unnormalized columns);
   - the strong-rule count for the next alpha.
2. Problems with rel. gap <= 0.9 tol stop. The others get working set = support + best candidates,
   sized `max(min(2|supp|, strong count + 16), |supp| + 16)`, doubling on later outer iterations.
3. Restricted problems are solved by a **CUDA Gram-CD kernel, one thread block per problem**
   (cupy/NVRTC, `_ws_common._KERNEL2_SRC`):
   - per group: one full epoch over the working set, then 25 epochs over the active set on a compacted,
     contiguous fp32 Gram of the active set, with register prefetch of the next row;
   - exact refresh of q = c - Gw from the fp64 Gram before each restricted-gap check;
   - Anderson extrapolation (5 iterates, 5 epochs apart, accepted only if the objective decreases).
   CD step sizes use the Gram diagonal, i.e. per-column norms; nothing assumes unit variance.
4. Grams come from a **union Gram** over every feature that has been in any working set, grown incrementally
   and shared by all targets, one per fold (G_f = X_U^T diag(m_f) X_U). If F·|U|² exceeds 30% of GPU memory,
   it falls back to **per-problem Grams** (B, K, K) via gather + bmm, in fp32 when tol >= 1e-5.
5. Warm starts down the path. The last check of alpha a is the first check of alpha a+1, so with the
   strong rule each alpha costs one full X pass (`n_full_matmul` = 50 for 50 alphas in every grid run).
6. `fit_cv`:
   - folds are extra row-masked problems in the same batch, with held-out MSE computed on the GPU per alpha;
   - **CV early stop**: a target's fold paths stop once its mean held-out MSE has stayed > 5% above
     its running minimum for 10 alphas;
   - the refit on all rows stops each target at its selected alpha.
7. Options:
   - `select="min"|"1se"`;
   - `l1_ratio` (elastic net, as an augmented Lasso: λ2 on the Gram diagonal and λ2·w subtracted on the
     support in the full check, so the gap stays exact);
   - `relax="cv"` (section 5);
   - `relax_gamma` (fixed-penalty relaxed Lasso, kept only for comparison);
   - `stream_x` (X in pinned host memory, streamed per full check);
   - `inner="fista"` (batched FISTA on dense restricted Grams, CD for stragglers).
   `fit_cv` also returns `fold_coefs` (K sparse (p, T) at the selected alpha) and `coef_lasso`.

**`torch_admm`**: batched ADMM with the matrix-inversion lemma, warm-started down the path, over-relaxation
1.6 and residual-balancing adaptive rho. One eigendecomposition XX^T = QΛQ^T gives
(XX^T + n·rho_b·I)^{-1} for every per-problem rho_b, so adaptive rho needs no refactorization. The Cholesky
variant needs n³/3 per rho change and one shared rho. Exact zeros come from z.

Correctness, all against sklearn at tol 1e-10:
- small problem (n = 300, p = 2000): max relative suboptimality 5e-11 to 1.4e-10, coef error ≤ 1.3e-4;
- elastic net vs `ElasticNet(alpha/rho, rho)`: suboptimality 5e-15;
- every grid run re-verifies all 50 × T gaps in float64 on the GPU;
- harness `syn-s` / `pca1k` / `unit-*` Task P rows reach the loose level, and tight (≤ 1e-6) at tol 1e-6.

## 2. Scaling (synthetic, `make_synthetic` recipe: rho = 0.5, k = 30, snr = 1; 50 alphas down to 1e-2 alpha_max)

Times are for the full path; gap verified in fp64 on every (alpha, target). `n_full_matmul` = 50 everywhere.

| n | p | T | tol | GPU | time | max gap | frac ≤1e-6 | ws max | union | full-corr s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e3 | 1e4 | 384 | 1e-4 | H200 | 14.1 s | 9.0e-5 | 0.59 | 1754 | 1e4 | <0.5 |
| 1e3 | 1e5 | 384 | 1e-4 | H200 | 21.2 s | 9.0e-5 | 0.56 | 1944 | 1e5 | <0.5 |
| 1e3 | 1e6 | 384 | 1e-4 | H200 | 27.2 s | 9.0e-5 | 0.50 | 1962 | 5.7e5 | 1.1 |
| 1e3 | 1e4 | 384 | 1e-6 | H200 | 40.1 s | 2.7e-7 | 1.00 | 2730 | 1e4 | <0.5 |
| 1e3 | 1e5 | 384 | 1e-6 | H200 | 67.3 s | 2.7e-7 | 1.00 | 2416 | 1e5 | <0.5 |
| 1e3 | 1e6 | 384 | 1e-6 | H200 | 90.8 s | 2.7e-7 | 1.00 | 3824 | 5.8e5 | 1.2 |
| 1e3 | 1e4 | 64 | 1e-4 | A100-80 | 14.1 s | 9.0e-5 | 0.59 | 1766 | 1e4 | <0.5 |
| 1e3 | 1e5 | 64 | 1e-4 | A100-80 | 21.2 s | 9.0e-5 | 0.53 | 1944 | 7.6e4 | <0.5 |
| 1e3 | 1e6 | 64 | 1e-4 | A100-80 | 24.5 s | 9.0e-5 | 0.50 | 1966 | 1.3e5 | <0.5 |
| 1e3 | 1e4 | 384 | 1e-4 | A100-80 | 22.4 s | 9.0e-5 | 0.59 | 1758 | 1e4 | <0.5 |
| 1e3 | 1e5 | 384 | 1e-4 | A100-80 | 32.6 s | 9.0e-5 | 0.56 | 1944 | 1e5 | <0.5 |
| 1e3 | 1e6 | 384 | 1e-4 | A100-80 | 39.4 s | 9.0e-5 | 0.50 | 1988 | 5.7e5 | 2.7 |
| 1e3 | 1e5 | 384 | 1e-6 | A100-80 | 118.1 s | 2.7e-7 | 1.00 | 2480 | 1e5 | <0.5 |
| 1e3 | 1e6 | 64 | 1e-4 | A100-80, `stream_x` | 40.1 s | 9.0e-5 | 0.50 | 1966 | 1.3e5 | 11.9 |
| 1e4 | 1e4 | 384 | 1e-4 | H200 | 119.5 s | 8.9e-5 | 0.81 | 7008 (= cap) | 1e4 | <0.5 |
| 1e4 | 1e5 | 64 | 1e-4 | A100-80 | 1655 s, **16% unconverged** | 6.4e-2 | – | 4960 (= cap) | 9.9e5 | 7.6 |
| 1e4 | 1e5, 1e6 | 384 | 1e-4 | H200 | **OOM** (per-problem Gram 384 × 7000²) | | | | | |

Time to 1e-4 vs 1e-6: about 2.8–3.3x (H200) and 3.6x (A100). The extra time is all inner CD in the
nnz ≈ n tail of the path.

**celer yardstick**, same data (harness `cpu_celer`, 32 processes, Xeon 8268, 32 targets timed then projected
to 384). At n = 1e3, T = 384:

| p | tol | celer, projected | torch_ws H200 | speedup | torch_ws A100 | speedup |
|---|---|---|---|---|---|---|
| 1e4 | 1e-4 | 511 s | 14.1 s | **36x** | 22.4 s | **23x** |
| 1e5 | 1e-4 | 1310 s | 21.2 s | **62x** | 32.6 s | **40x** |
| 1e4 | 1e-6 | 644 s | 40.1 s | **16x** | – | – |

For a single-process baseline: celer took 12.7 s/target at p = 2e4, tol 1e-4.

**Real tiers, harness.** H200 rows at e73ac90 / 99f72ff, A100 at 99f72ff. CPU references are from the bench-harness rows.

| tier | task | torch_ws H200 | torch_ws A100 | CPU reference |
|---|---|---|---|---|
| unit-100k (64 ch) | P, tol 1e-4 | 38.3 s | 41.8 s | – |
| unit-100k | P, tol 1e-6 | 171.8 s (round 1) | 191.2 s (round 1) | – |
| unit-100k | CV | **11.5 s**, R2_rel 0.3257, nnz 120.5 | 13.7 s, 0.3257 | celer 403 s (32 thr), 0.3257; RidgeCV 2.8 s, 0.347 |
| unit-100k-raw | CV | 5.4 s, 0.3159 | 6.1 s | celer 948 s, 0.3159; RidgeCV 3.7 s, 0.352 |
| unit-800k (64 ch, layer1) | CV | 20.1 s, R2_rel 0.0326 | see section 5 | RidgeCV 10.3 s, 0.125 |
| unit-800k-383 | CV | 21.7 s, R2_rel 0.405 | 29.4 s | round 1: 75.8 s H200 / 239.7 s A100 |
| unit-800k-raw | CV | 4.1 s, R2_rel 0.003 | 5.8 s | celer 1815 s, 0.003 |

## 3. Where the time goes (profiles from `solver_info.timings`)

Inner CD is 85–98% of wall time on every tier. The screening/KKT pass (`full_corr`) is 1–5%.

unit-800k-383 CV, round 1 on H200, 74.7 s for the 5 fold paths:

| phase | time |
|---|---|
| inner CD | 58.5 s |
| per-problem Gram build | 4.3 s |
| full correlation (screening + KKT, 1915 problems × 7.8e5 features × ~50 checks) | 3.8 s |
| WS build | 2.8 s |
| residuals | 2.1 s |
| held-out MSE | 1.3 s |
| sparse extraction | 0.3 s |
| refit | 0.7 s |

Union of working sets: 5.6e5 features. Per alpha, ws_max rises 432 → 1628 and nnz_max 158 → 832 (n = 822).

Inner CD dominates because of the **path tail**. At 1e-2 alpha_max, nnz reaches ≈ n on every tier
(pca1k nnz_max 790 of n = 822, unit-800k 832, synthetic 930–1000). The restricted problems there are nearly
singular and need hundreds to thousands of sequential CD epochs. Task CV never needs that tail: the selected
alpha index has median 13.5 / max 28 on unit-100k and median 5 / max 25 on unit-800k-383. Hence the CV early
stop (section 4).

Small-p overhead (coordinator item 3) is not Python, transfers or KKT passes. On pca1k, 10.19 of 10.4 s is
inner CD (K reaches all 1000 features at nnz ≈ 790). With CV early stopping pca1k CV takes 1.0 s on A100,
down from 25.6 s.

## 4. Round-2 speed changes, with numbers

| change | before → after | selection identical? |
|---|---|---|
| **fold batching always on** (`cv_batch = K`; the old auto rule fell back to sequential folds on 80 GB) | A100 unit-100k CV 170.8 s → 13.7 s; unit-800k-383 239.7 s → 29.4 s | yes |
| **CV early stop** (10 alphas, 5%) | unit-100k CV 33.5 s → 6.5 s (H200, warm process); unit-800k 42.5 → 15.2 s; pca1k 25.6 → 1.0 s (A100); syn-s 13.4 → 1.7 s | yes: best_idx agrees on 100% of channels on all four tiers, identical test R2 |
| **kernel warm-up** of both Gram dtypes in every call | fp32 per-problem-Gram kernels were compiled inside the timed CV region (harness 11.5 s vs 6.5 s in a warm process); now compiled by the harness warm-up | – |
| kernel v2 (compacted active Gram + prefetch + spaced Anderson) vs v1 | n = 1e3, p = 2e4, T = 384, 1e-4: 63.0 → 18.4 s (H200), 89.8 → 62.4 s (A100) | – |
| Anderson in v1 | 16.4 → 4.2 s (n = 300, p = 2000, T = 8, 1e-6) | – |
| batched Newton polish on the active set (Cholesky) | 97.4 s with, 63.0 s without (T = 384): **off** by default | – |
| ws rule 2|supp| vs strong-rule set | p = 1e5, T = 64: 18.9 vs 16.8 s, but ws 1936 vs 3633 and less memory; kept 2|supp| | – |
| `inner="fista"` (batched FISTA on dense restricted Grams) | unit-100k CV on A100: 9.0 s with early stop (CD: 13.7 s in the harness; same machine, different process state, so not a clean A/B) | yes |

**A100 vs H200 (coordinator item 2).** The 5x gap was a config bug, not hardware or contention. The old
`cv_batch="auto"` rule estimated the union-Gram memory as K·30000²·8 B > 0.35 × 80 GB and solved the 5 folds
one after another on A100, while H200 (141 GB) batched them. Round-1 A100 fold logs show five sequential
32–37 s fold solves (`cv_batch` = 1). Fixed in b9f84c1 and re-run on the committed code: unit-100k CV is
13.7 s A100 vs 11.5 s H200, unit-800k-383 CV 29.4 vs 21.7 s.

The remaining 1.2–1.6x is the CD kernel. It is latency-bound (one thread block per problem, two
`__syncthreads` per coordinate). The A100 has fewer SMs (108 vs 132), a lower clock and less shared memory
per SM (164 vs 228 KB). The last point means 2 instead of 3 resident blocks at K ≈ 2000, so T = 384 runs in two
waves. Task P on synthetic shows the same: 14.1 vs 22.4 s at p = 1e4, T = 384.

## 5. Sparse variants (round 2)

Setup:
- `scripts/dev_ws_variants.py`, tol 1e-4, H200 for unit-100k (64 ch, 20 reliable), A100 for unit-800k
  (64 ch, 20 reliable).
- R2 is mean test R2 over reliable channels.
- Frontier: reliable R2 at k nonzeros, interpolated in log k along the full-data path, read at
  k = 10 / 30 / 100 / 300. k = 1000 is beyond the path's nnz on these tiers (nnz ≤ n = 822).
- Jaccard: mean pairwise support overlap across the 5 folds at the selected alpha.
- The CV time column is from the dev script. The first variant in a process includes kernel compilation
  (lasso-min 11.4 s); the harness rows in section 2 are the timed numbers.

**unit-100k** (RidgeCV reliable R2: 0.347).

| variant | select | CV time | R2_rel | nnz (median, rel.) | fold Jaccard | R2@10 | R2@30 | R2@100 | R2@300 |
|---|---|---|---|---|---|---|---|---|---|
| Lasso | min | 11.4 s* | **0.326** | 238 | 0.148 | 0.125 | 0.220 | 0.312 | 0.320 |
| Lasso | 1se | 6.2 s | 0.294 | 77 | 0.212 | | | | |
| elastic net rho = 0.5 | min | 6.6 s | **0.328** | 280 | 0.156 | 0.096 | 0.196 | 0.306 | **0.327** |
| elastic net rho = 0.5 | 1se | 5.4 s | 0.297 | 90 | 0.226 | | | | |
| elastic net rho = 0.9 | min | 6.0 s | 0.327 | 242 | 0.150 | 0.122 | 0.218 | 0.311 | 0.321 |
| **relaxed, CV-chosen ridge** | min | 12.0 s (+3.4 s refit CV) | 0.313 | 158 | 0.180 | **0.221** | **0.282** | **0.319** | 0.312 |
| **relaxed, CV-chosen ridge** | 1se | 9.3 s | 0.276 | **29** | **0.241** | | | | |
| relaxed, fixed gamma 0.01 | min | 2.4 s | 0.241 | 52 | 0.227 | 0.200 | 0.243 | 0.240 | 0.155 |
| relaxed, fixed gamma 0.1 | min | 10.3 s | 0.248 | 83 | 0.205 | 0.208 | 0.254 | 0.262 | 0.202 |

\* includes kernel compilation (first variant in the process).

- Relaxed CV, same run: plain-Lasso R2 is 0.326 / nnz 238 (min) and 0.294 / 77 (1se).
- Chosen penalties (min rule): 17% of channels keep the Lasso coefficients; most pick 1x–10x the mean diagonal.
- The relaxed-CV frontier (0.221 / 0.282 / 0.319 at k = 10 / 30 / 100) reproduces the gpu-prox
  matched-sparsity study (0.214 / 0.280 / 0.332).

**unit-800k, 64 ch** (RidgeCV reliable R2: 0.125). This is a hard, low-SNR tier for Lasso: all variants are below 0.07.

| variant | select | CV time (A100) | R2_rel | nnz | fold Jaccard | R2@10 | R2@30 | R2@100 | R2@300 |
|---|---|---|---|---|---|---|---|---|---|
| Lasso | min | 29.5 s | 0.033 | 152 | 0.093 | 0.014 | 0.035 | -0.056 | 0.042 |
| Lasso | 1se | 19.0 s | -0.068 | 0 | 0.120 | | | | |
| elastic net rho = 0.5 | min | 20.7 s | **0.059** | 187 | 0.102 | 0.011 | 0.029 | 0.038 | 0.062 |
| elastic net rho = 0.9 | min | 21.8 s | 0.058 | 157 | 0.097 | 0.013 | 0.035 | 0.035 | 0.064 |
| relaxed, fixed gamma 0.01 | min | 37.4 s | -0.211 | 5 | 0.137 | -0.017 | -1.36 | -0.67 | -0.19 |
| relaxed, CV-chosen ridge | min/1se | RELAXCV_U800K | | | | | | | |

Harness rows, H200 at e73ac90, tags `ws-r2-{lasso,relaxcv}-{min,1se}`, CV: HARNESS5_TABLE

**Fixed-gamma relaxed Lasso on unit-800k: why H200 (-0.75) and A100 (-0.21) disagree.** That estimator is
numerically unstable, so it amplifies run-to-run differences. The support refit solves
(G_SS + 0.01·mean(diag)·I) β = c on layer1 units, where 57% of the entries are exactly zero. Many units in a
support are nearly collinear on the training images, so the system is close to singular at gamma = 0.01. β then
takes huge values that blow up on test images (R2 -1.36 at k = 30). Which units enter the support near ties
depends on non-deterministic GPU reduction order (atomic `index_put_` and `topk` over chunks), which differs
between GPUs and runs. Repeats on the same H200: RELAX_REPEAT. The CV-chosen penalty removes the issue: the
grid reaches 100x the diagonal, and the plain Lasso coefficients are a candidate. Recommendation: drop the
fixed-gamma variant.

**Not built: (d) adaptive / reweighted L1 and MCP.** Both fit the same kernel: a per-feature penalty weight
in the soft-threshold, plus screening on c_j / w_j. They were not done in this round.

**API (round-2 item 2).** No `api.py` change is needed. `fit_cv` is a solver-side protocol documented in
`lassobench/runner.py`. The smallest change is one optional key in that docstring:
`fold_coefs: list[K] of scipy sparse (p, T)`, each fold's coefficients at the selected alpha per target.
`torch_ws.fit_cv` already returns it, plus `coef_lasso` (the plain-Lasso model from the same run, when
`relax="cv"`). `default_fit_cv` could fill `fold_coefs` the same way.

## 6. Design questions (round 1), answered with numbers

- **(i) Inner solver.** One-block-per-problem Gram CD on the GPU, with Anderson and a compacted active Gram,
  is the solver to keep.
  - FISTA on dense restricted Grams: correct, comparable on unit-100k CV, slower on small synthetic
    problems (3.4 vs 2.1 s); kept as an option.
  - Handing subproblems to the CPU: not built. celer's whole solve is 12.7 s/target at p = 2e4, while the
    GPU does 384 targets in 18 s, so per-problem CPU work cannot compete without hundreds of cores.
  - Shared union Gram vs per-problem Grams: the union wins while F·U² fits (no rebuilds). Per-problem
    Grams cost 1–6 s per path at p ≥ 1e5. The union reaches ≈ p at p ≤ 1e5 because 384 targets × 2|supp|
    working sets cover nearly everything.
- **(ii) Working-set sizes.** ws_max ≈ 2·nnz_max ≈ 2n at the path end: 1750–1990 at n = 1e3 (tol 1e-4),
  up to 3800 at tol 1e-6. On unit tiers, ws_max 1628 for nnz 832. The union covers 1e4 of 1e4 features,
  1e5 of 1e5, and 5.7e5 of 1e6 (T = 384), or 1.3e5 at T = 64.
- **(iii) Extrapolation.** Primal Anderson inside CD: 4x in v1, 2x in v2. Dual extrapolation for the
  outer gap was not needed: the outer loop already takes ~1 full pass per alpha, and the inner tolerance
  is set tight.
- **(iv) Fold batching.** Row masks as problems, with per-fold union Grams as masked matmuls on the
  incremental columns. syn-s CV went from 56.7 s (5+1 `fit_path` calls) to 14.0 s, identical selections
  and coefficients equal to 6e-9 (A100). With the always-on batching fix, A100 unit-100k went from 171 s to
  13.7 s.
- **(v) X larger than GPU memory.** `stream_x=True`: pinned host column blocks with side-stream prefetch.
  p = 1e6, n = 1e3 (4 GB) took 40.1 vs 24.5 s resident on A100; full-corr went from 0.5 s to 11.9 s for
  50 passes (~17 GB/s). At n = 1e4 (40 GB per pass) that is ~2.4 s per alpha, ~2 min per path, so
  workable. The binding limit at n = 1e4 is the inner solver, not X (section 7).

## 7. Failure modes

- **n = 1e4, deep path.** nnz ≈ 5000 at 1e-2 alpha_max exceeds the kernel's shared-memory working-set cap
  (7008 on H200, 4960 on A100). Per-problem Grams (384 × 7000²) do not fit:
  - H200 OOM at p = 1e5 and 1e6;
  - A100 p = 1e5, T = 64: 1655 s, 16% of problems unconverged.
  Fixes: chunk the kernel launch over problems with per-chunk Grams, and a global-memory fallback for
  K > cap. Task CV largely avoids the tail through early stopping.
- **ADMM.** On unit-100k at 1e-4 the A100 run stopped at max_iter with gap 1.4e-4 (139 s).
- **NFS staleness.** Compute nodes can see a half-updated `.py` for a few seconds after an edit. Two round-2
  harness rows failed with an `ImportError` this way, and their re-runs are at e73ac90. Don't edit solver
  files while harness jobs are starting.

## 8. ADMM: drop

| case | torch_admm | torch_ws |
|---|---|---|
| pca1k, P, 1e-4, H200 | 7.8 s | 10.5 s |
| unit-100k, P, 1e-4, H200 | 50.8 s | 38.6 s |
| unit-100k, P, 1e-4, A100 | 139 s, did not reach 1e-4 | 46.7 s |
| unit-100k, P, 1e-3, H200 | 17.5 s | – |
| synthetic p = 1e4, T = 384, 1e-4, A100 | 13.5 s, 99.7% converged | 22.4 s |
| synthetic p = 1e5, T = 384, 1e-4, A100 | 160 s | 32.6 s |
| synthetic p = 1e5, T = 384, 1e-3, A100 | 81.6 s | – |

ADMM wins only at small p (≤ 1e4) and loose tolerance, a regime that `torch_fista` already covers (pca1k
P 1.3 s). Each iteration is two O(n p B) matmuls, against about one per *alpha* for `torch_ws`, so it falls
behind linearly in p and cannot reach tight gaps in fp32. No CV rows: it is not worth building `fit_cv` for it.

## 9. Keep / drop

| idea | call | evidence |
|---|---|---|
| GPU working set + one-block-per-problem Gram CD (v2) | **keep, default** | sections 2 and 4 |
| CV early stopping | **keep, default** | 3–25x faster CV, identical selections on 4 tiers |
| fold batching (row masks) | **keep, default** | 4x syn-s, fixes the A100 5x |
| union Gram with per-problem fallback | keep | |
| Anderson in CD | keep | 2–4x |
| batched Newton polish | drop (off) | slower at T = 384 |
| elastic net (l1_ratio) | keep as option | best min-CV R2 on both unit tiers (0.328, 0.059); not sparser |
| relaxed Lasso, CV-chosen penalty | **keep; recommended for the sparse readout** | frontier +0.06–0.10 at k ≤ 30; 1se gives 29 nonzeros at R2 0.276 and the highest fold Jaccard (0.241) |
| relaxed Lasso, fixed gamma | drop | unstable (section 5) |
| 1-SE rule | report both | about 3x fewer nonzeros at -0.03 R2 on unit-100k; on unit-800k it selects the empty model |
| `inner="fista"` | keep as option | comparable; may suit small B |
| `stream_x` | keep as option | needed for X > GPU memory |
| torch_admm | **drop** | section 8 |
| SSNAL | not built | the inner solve in the nnz ≈ n tail is the only place it would help, and early stopping removed most of that cost for CV |

Next wins, in order:
1. Kernel chunking plus a global-memory fallback, so n = 1e4 runs to the path end.
2. Adaptive L1 / MCP in the kernel, for sparser supports at equal R2.
3. A per-target alpha floor for Task P: stop each path where nnz approaches n.
