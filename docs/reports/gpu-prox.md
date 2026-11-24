# gpu-prox report: batched FISTA and deep-learning-optimizer Lasso on GPU

Session `gpu-prox`. Files: `lassobench/solvers/{torch_fista,torch_proxgrad,_torch_common}.py`,
`scripts/dev_prox_*.py`, `slurm/prox_*.sh`. Raw run rows: `STORE/results/prox/*.jsonl` (synthetic, via
`scripts/dev_prox_scale.py`), harness rows tagged `gpu-prox` in `tables/results.jsonl`. Every number below is
copied from one of those files.

## Verdict (TL;DR)

- **Full-p FISTA is a small-p solver only. Drop it for the unit-basis tiers.** On real `unit-100k`, FISTA needs
  553 s (H200) and still leaves 2081/3200 problems above gap 1e-4; strong-rule screening brings that to 371 s,
  still 1946 unconverged. `torch_ws` solves the same path to 1e-4 in 38.6 s. The cause is conditioning: real unit
  features have L = lambda_max(X X^T/n) = 3842 (synthetic: ~110), so the global 1/L step is ~3800x smaller than the
  unit coordinate curvature. Restricted to a single target's support L drops to 4-21, but restricted to the
  union over 64 targets (all a batched full-X FISTA can exploit) it stays at 315-891 (table 5). So the only way
  to rescue FISTA is per-target working sets, which is exactly `torch_ws` (it already runs batched FISTA on
  restricted per-problem Grams). FISTA's role is the inner solver, not the outer one.
- **Where FISTA does win:** dense-ish, small-p, well-conditioned problems. `pca1k` full CV in 2.4 s (H200) vs
  sklearn LassoCV 112 s (old) / celer P alone 13 s; synthetic p <= 1e5 paths in 4-13 s on H200. Batched CV with
  per-fold row masks on a shared X costs about the same as a single path (syn-s: CV 8.5 s vs P 8.3 s).
- **Gradient optimizers (the user's question):** with a final full-batch prox step they produce exactly sparse,
  *predictively competitive* solutions on `unit-100k`, but they are not faster than the working-set solver:
  - Hadamard `w = u*v` + Adam (1000 epochs) + final prox step reproduces the exact Lasso CV result: reliable-channel
    test R2 0.3257 and nnz median 121.5 vs exact (torch_ws / celer) 0.3257 and 120.5. Cost 219 s vs 33 s for torch_ws (H200).
  - Prox-SGD (= ISTA, 1000 epochs) and prox-Adam (300 epochs) are far from the Lasso optimum (median rel gap
    0.12-0.19) but give **higher** test R2 on reliable channels (0.367 / 0.361) than exact Lasso (0.326) and
    RidgeCV (0.347), at 5-11x more nonzeros (nnz median 1368 / 568 vs 120). That is early-stopping
    regularization, not a better Lasso solver; worth a note to the coordinator as a modelling result.
  - Minibatch prox-Adam (bs 128, 100 epochs): R2 0.338, nnz 299, 99 s for CV; it is the only route when X does
    not fit on the GPU, and streaming X from pinned host memory costs 1.85x (P: 21.8 s vs 11.8 s).
- **Round 2 (section 11): at matched sparsity no gradient method beats relaxed exact Lasso** (k <= 100 all
  differences <= +0.004, mostly significantly negative); the early-stopped R2 edge is a density effect.
- **Recommendation:** keep `torch_fista` only as (a) the small-p / PCA-tier solver and (b) a reference
  implementation of the batched-CV-by-masks trick (worth porting to `torch_ws` if it does not do it already).
  Keep `torch_hadamard_adam` as the honest answer to "can a plain DL optimizer do Lasso": yes, it matches the
  exact solution, at ~6.6x the cost of the working-set solver. Drop union screening, CUDA graphs and TF32/bf16
  inside FISTA. Details and numbers follow.

## 1. What was built

`torch_fista` (`TorchFISTA`): a pool of "slots", each one (target, CV fold, contiguous segment of the alpha
path), stored slot-major as `W (B, p)` so the live batch is a contiguous prefix. Per iteration: `G = R_z @ X`
and `Xw = W @ X^T` (two GEMMs), plus one fused Triton kernel (momentum + gradient step + soft threshold +
gradient-restart dot product; z is never materialized because the gradient only needs `Xz`, which lives in
n-space). Every `check_every` iterations it computes the duality gap per slot (one extra GEMM). Converged slots
emit their sparse column to the host, advance to the next alpha (warm start, momentum reset), and immediately
re-check the gap with the same residual (free). Finished slots are refilled from a pending queue or compacted
away. `segments` interpolates between pure warm-start path (1) and all alphas at once from zero (A). `fit_cv`
solves all K folds and the refit in one batch, with 0/1 row masks per fold on the shared X (no X copies,
per-fold Lipschitz by masked power iteration). Options: `dtype` (float32/float64/bfloat16), `tf32`, `restart`,
`screen` (union strong rule, Lipschitz recomputed on the screened set), `cuda_graph`, `tol_margin`.

`torch_proxgrad` (`TorchProxGrad` and presets `torch_prox_adam`, `torch_prox_adam_mb`, `torch_prox_sgd`,
`torch_subgrad_adam`, `torch_hadamard_adam`): all T x A problems at once (waves if memory requires), manual
gradients, full batch or minibatch over images, optional X streamed from pinned host memory. Methods: prox-SGD
(ISTA at lr=1/L), prox-Adam (prox in Adam's diagonal metric, exact zeros), subgradient Adam, Hadamard u*v Adam.
`finalize='prox'` applies one full-batch ISTA step at the end (exact zeros; counted in the time).

## 2. Correctness

- vs sklearn `Lasso(fit_intercept=False, tol=1e-8)` (`scripts/dev_prox_check.py`, n=100, p=500/2000, T=2-3,
  A=5-8, float64): worst relative objective difference 1.7e-15 (no screening), 3.9e-14 (screening), 8e-16
  (screening + reduced-set Lipschitz). In float32 at tol 1e-6: 4e-9.
- Harness `syn-s` Task P vs the cached float64 reference: float64 tol 1e-8 gives gap_rel_max 0 and
  subopt_rel_max 0 (converged_level `tight`, 24.5 s H200); CV support Jaccard vs reference 0.999.
  `pca1k`: float64 tol 1e-8 converged tight, subopt_rel_max 1e-6.
- Note: CSC output is float32 (contract), which by itself puts a ~1e-7 floor on the recomputed relative gap.

## 3. Synthetic scaling (make_synthetic model, A=50, eps=1e-2, Task P, `STORE/results/prox/scale.jsonl`)

Time in seconds, wall clock including transfer and Lipschitz. "iters" = sequential FISTA iterations for the
whole path; "med" = median iterations per (alpha, target) problem. Default config (fp32, segments=1, restart,
fused) unless noted.

| n | p | T | GPU | config | time s | iters | med | unconverged |
|---|---|---|---|---|---|---|---|---|
| 1e3 | 1e4 | 64 | H200 | tol 1e-4 | 6.0 | 11470 | 200 | 0 |
| 1e3 | 1e4 | 64 | H200 | tol 1e-4, segments 5 | 4.0 | 4440 | 200 | 0 |
| 1e3 | 1e4 | 64 | H200 | tol 1e-6, fp64 | 12.0 | 31880 | 330 | 0 |
| 1e3 | 1e4 | 64 | A100-80GB | tol 1e-4 | 6.0 | 11670 | 190 | 0 |
| 1e3 | 1e4 | 64 | A100-80GB | tol 1e-4, segments 5 | 3.3 | 4440 | 200 | 0 |
| 1e3 | 1e4 | 64 | A100-80GB | tol 1e-6, fp64 | 16.0 | 33780 | 320 | 0 |
| 1e3 | 1e5 | 64 | H200 | tol 1e-4 | 12.8 | 16390 | 360 | 0 |
| 1e3 | 1e5 | 64 | H200 | tol 1e-4, segments 5 | 11.6 | 5160 | 360 | 0 |
| 1e3 | 1e5 | 64 | H200 | tol 1e-6, fp64 | 59.6 | 61190 | 810 | 0 |
| 1e3 | 1e5 | 64 | A100-80GB | tol 1e-4 | 31.8 | 16560 | 360 | 0 |
| 1e3 | 1e5 | 64 | A100-80GB | tol 1e-6, fp64 | 147.9 | 64690 | 830 | 0 |
| 1e3 | 1e5 | 384 | H200 | tol 1e-4 | 53.7 | 16790 | 360 | 0 |
| 1e3 | 1e5 | 384 | H200 | tol 1e-6, fp64 | 268.5 | 69060 | 820 | 0 |
| 1e3 | 1e5 | 384 | A100-80GB | tol 1e-4 (*) | 222.0 | 46910 | 480 | 42 |
| 1e3 | 1e5 | 384 | A100-80GB | tol 1e-6, fp64 | 826.0 | 76870 | 900 | 0 |
| 1e4 | 1e5 | 64 | H200 | tol 1e-4 (*) | 112.7 | 29610 | 70 | 4 |
| 1e4 | 1e5 | 64 | H200 | tol 1e-6, fp64 | 133.2 | 23330 | 150 | 0 |
| 1e4 | 1e5 | 64 | A100-80GB | tol 1e-4 (*) | 226.6 | 19940 | 70 | 4 |
| 1e4 | 1e5 | 64 | A100-80GB | tol 1e-6, fp64 | 339.1 | 22600 | 150 | 0 |
| 1e3 | 1e6 | 64 | H200 | tol 1e-4 (*) | 2187 | 336280 | 10000 | 1602 |
| 1e3 | 1e6 | 64 | H200 | tol 1e-4, segments 5 (*) | 2005 | 100000 | 3005 | 1579 |
| 1e3 | 1e6 | 64 | A100-80GB | tol 1e-4 (*) | 5010 | 319430 | 3160 | 1594 |

(*) run with an internal stopping margin of 0.5 x tol (since changed to 0.8): fp32 cannot always reach 5e-5, so
these rows include problems that ran to max_iter=10000 (this inflates the A100 T=384 and n=1e4 rows). At p=1e6
the failures are genuine slow convergence (gap max 5.4e-4): L grows ~ p/n for this model.
n=1e4, p=1e6 and syn-xl were cancelled: at ~0.04-0.25 s per iteration and >1e5 iterations they cannot finish.

Time to loose vs tight: at n=1e3, p=1e5, T=64 on H200, rel gap 1e-4 takes 16390 iterations / 12.8 s, 1e-6
(fp64) takes 61190 iterations / 59.6 s (3.7x iterations; per problem median 360 -> 810).

## 4. What mattered inside FISTA (H200, n=1e3, p=1e5, T=64, tol 1e-4 unless noted; `ablate*_p1e5`, `graph_p1e5`, `screen_p1e5`)

| change | time s | vs base 12.9 s | notes |
|---|---|---|---|
| base (fp32, warm-start path, restart, fused prox) | 12.9 | - | 16390 iters, 0.79 ms/iter |
| unfused prox (plain torch ops) | 16.2 | +25% | fused Triton kernel: momentum+prox+restart in one pass |
| no gradient restart | 14.2 | +10% | 1.10M vs 0.96M slot-iterations |
| segments 5 | 11.6 | -10% | best; more parallel slots, cold-start cost small |
| segments 10 | 12.1 | -6% | |
| segments 50 (all alphas at once, cold) | 21.5 | +67% | 2.4x more slot-iterations than warm start |
| check_every 5 / 25 | 13.9 / 12.0 | +8% / -7% | gap check is one extra GEMM |
| float64 | 17.5 | +36% | H200 FP64 GEMM is as fast as FP32 (50 TFLOPS, `prox_mmbench_h200.log`) |
| TF32 matmul | 144.7 | never converges | gap floor: median 4e-4, max 2.4e-3; 2101/3200 hit max_iter |
| bf16 X | 182.6 | never converges | gap floor: median 3.2e-3, max 1.5e-2 |
| CUDA graphs (6-iteration capture) | 13.6 | +5% | iterations are GEMM-bound, not launch-bound |
| union strong-rule screening | 13.0-23.0 | 0 to +78% | union over 64 targets keeps 79% of p (q_mean 78.6k) |
| tol 1e-6 fp64, segments 5/10 | 52.7 / 51.5 | vs 59.6 | segments help more at tight tol |

Profile (synced timers, so absolute time is inflated to 17.1 s; base config): mm_fwd 32%, mm_grad 29% (fp32 GEMMs
at ~35-50 TFLOPS on H200), prox 17%, momentum 12%, gap 5%, extract 5%. Microbenchmark (`scripts/dev_prox_mmbench.py`, n=822, p=1e5, H200): per-column
GEMM cost fp32 6.9 us, TF32 3.4 us, bf16 1.4 us, fp64 5.7 us at B=384; B=16 is latency-bound (25 us/column).

Precision summary: fp32 reaches rel gap 1e-4 reliably and ~1e-5 to 5e-5 at best (problem dependent); 1e-6 needs
float64, which on H200 costs only ~1.4x per iteration. TF32/bf16 cannot reach 1e-4 at all with the full-precision
gap check, although the test R2 at those floors is unchanged (0.3543 vs 0.3543), i.e. they are fine as
*approximate* solvers but not for Task P.

## 5. Real tiers (harness rows, tag `gpu-prox`, tol 1e-4)

Task P (single path, 50 alphas):

| tier | solver | GPU | time s | gap_rel_median / max | frac problems <= 1e-4 | nnz median |
|---|---|---|---|---|---|---|
| unit-100k | torch_fista | H200 | 552.8 | 1.1e-4 / 8.3e-4 | 0.47 | 508.5 |
| unit-100k | torch_fista screen | H200 | 371.1 | 1.8e-4 / 3.0e-3 | 0.41 | 514.5 |
| unit-100k | torch_ws (gpu-workingset) | H200 | 38.6 | - / 1e-4 | 1.00 | 508.5 |
| pca1k | torch_fista | H200 | 1.27 | 1.6e-5 / 1e-4 | 1.00 | 471.5 |
| pca1k | cpu_sklearn_cd (32 thr) | CPU | 4.74 | 6.4e-5 / 1e-4 | 1.00 | 471.5 |
| pca1k | cpu_celer (32 thr) | CPU | 12.99 | 1.8e-5 / 4.1e-4 | 0.94 | 471.5 |

unit-800k (T=383, p=779947): see section 8 (rows still running when this was written; recorded as they land).

Why FISTA fails on unit features (`scripts/dev_prox_spectrum.py unit-100k`, supports from torch_ws at tol 1e-4):

| | features | L = lambda_max(X_S^T X_S / n) |
|---|---|---|
| all of X | 100352 | 3804 (next: 2677, 1856, 1436; 10th 793; 50th 295; median 77) |
| union of supports over 64 targets, alpha idx 5 / 15 / 25 / 35 / 49 | 1932 / 12996 / 23968 / 29732 / 32522 | 59 / 315 / 599 / 787 / 891 |
| single-target support (median), same alphas | 19 / 187 / 500 / 686 / 788 | 4.0 / 9.5 / 14.5 / 17.0 / 20.8 |

The spectrum decays slowly (no single dominant mode to deflate), and the batched union of supports is a third
of p with L ~ 900. Per-target active sets have ~40-200x smaller L than anything a shared-X batched method can
use. This is the quantitative reason the working-set solver wins and full-p FISTA should be dropped there.

## 6. Gradient-based optimizers vs exact

### unit-100k (harness, H200; exact reference = torch_ws / celer CV: reliable R2 0.3257, nnz median 120.5; RidgeCV 0.3468)

| solver (config) | P time s | P gap_rel median / max | P nnz median | CV time s | CV R2 reliable | CV nnz median |
|---|---|---|---|---|---|---|
| prox_adam (300 ep, lr 1e-2) | 10.7 | 0.19 / 2.6 | 3730 | 81.3 | 0.361 | 568 |
| prox_adam (1000 ep, lr 1e-2) | 32.2 | 0.047 / 0.58 | 820 | 191.3 | 0.330 | 150 |
| prox_sgd = ISTA (1000 ep, 1/L) | 25.4 | 0.12 / 0.26 | 4893 | 154.6 | 0.367 | 1368 |
| prox_adam minibatch 128 (100 ep) | 11.8 | 0.85 / 6.3 | 9518 | 99.1 | 0.338 | 299 |
| subgradient Adam + final prox (1000 ep) | 30.9 | 0.070 / 0.81 | 1665 | 185.3 | 0.333 | 955 |
| Hadamard u*v Adam + final prox (1000 ep) | 40.2 | 4.4e-4 / 1.8e-2 | 526.5 | 219.4 | **0.3257** | **121.5** |
| exact: torch_ws | 38.6 | <= 1e-4 | 508.5 | 33.2 | 0.3257 | 120.5 |

Objective suboptimality vs reference is not available on unit-100k yet (no cached reference when these ran);
the gap column is the certificate. On synthetic data (n=1e3, p=1e5, T=64, `pg2_p1e5.jsonl`, reference = fp64
FISTA tol 1e-6; note: those wall times were measured while another job shared the node, so use them only
relative to each other):

| solver | rel gap median | subopt median / max | R2 at reference's oracle alpha (ref 0.354) | nnz there (ref 208) |
|---|---|---|---|---|
| prox_adam 300 ep lr 1e-2 (A100 run) | 0.124 | 0.034 / 0.20 | 0.268 | 292 |
| prox_adam 300 ep lr 3e-2 | 0.069 | 0.0064 / 0.23 | 0.343 | 209 |
| prox_adam 300 ep lr 1e-1 | 0.058 | 0.0074 / 0.40 | 0.354 | 208 |
| prox_adam 1000 ep lr 1e-2 | 0.038 | 0.0018 / 0.063 | 0.345 | 210 |
| prox_sgd (ISTA) 300 ep | 0.12 | 0.062 / 0.35 | 0.271 | 376 |
| prox_adam minibatch 128, 100 ep | 0.094 | 0.028 / 0.66 | 0.329 | 262 |
| prox_adam minibatch 32, 30 ep | 0.32 | 0.21 / 13.9 | 0.180 | 1340 |
| subgradient Adam + prox, 1000 ep | 0.048 | 0.0041 / 0.082 | 0.347 | 435 |
| subgradient Adam, no final prox | - | - | 0.230 | 1e5 (no zeros at all) |
| Hadamard Adam + prox, 1000 ep | 0.0031 | 2.1e-6 / 2.3e-3 | 0.354 | 209 |

Takeaways:
- Exact zeros require a prox step. Subgradient Adam alone leaves every coefficient nonzero (nnz 1e5); the final
  full-batch prox step (one extra GEMM pair) fixes sparsity but not the optimization error.
- Adam is very sensitive to lr because the per-coordinate step is ~lr regardless of the gradient scale; on raw
  (non z-scored) features this needs retuning per tier, unlike FISTA/ISTA which use 1/L.
- Hadamard u*v is the only DL-style method that gets close to the Lasso optimum (median gap 3e-3 synthetic,
  4.4e-4 unit-100k) and it matches the exact support size. It is still 1000 epochs of two dense GEMMs over all
  T*A problems: on unit-100k Task P it takes about as long as torch_ws (40.2 s vs 38.6 s) without reaching 1e-4
  (41% of problems do), and Task CV is 6.6x slower (219 s vs 33 s) because the default protocol refits 6 times.
- Minibatch costs: per epoch the same flops as full batch, but more steps and noisy prox; at bs 128 it lands
  near full-batch prox-Adam quality in ~1/3 the epochs; bs 32 is clearly worse. Streaming X from host memory
  (only route when X does not fit on the GPU) doubled the time (21.8 s vs 11.8 s P on unit-100k).
- The higher test R2 of ISTA / prox-Adam at loose convergence is real on this tier (0.36-0.37 vs Lasso 0.326 and
  Ridge 0.347) but comes with 5-11x more nonzeros; it is an early-stopping/implicit-ridge effect and argues for
  trying elastic-net or relaxed Lasso, not for these optimizers as Lasso solvers.

## 7. Failure modes

- Ill-conditioning (real unit features, L ~ 3800; synthetic p=1e6, L ~ p/n): FISTA iterations scale with sqrt(L)
  per problem and every iteration pays a full-p GEMM. 47% of unit-100k problems unconverged after 10k iterations.
- Small alpha with correlated features is the slow end: per-problem median iterations rise along the path
  (synthetic p=1e5, H200, tol 1e-4: 10 at alpha_max, ~100 by alpha index 8, peak ~480 around index 28, ~340 at
  0.01 alpha_max) and the fp64 tol-1e-6 maximum is
  3850-6530 iterations.
- fp32 precision floor at ~1e-5 to 5e-5 relative gap; TF32 ~4e-4; bf16 ~3e-3. A stopping target below the floor
  burns max_iter (seen with tol_margin 0.5).
- Union screening over a batch of correlated targets keeps most of p and churns (17755 rebuilds on unit-100k).
- `torch.nonzero` cannot handle > 2^31 elements: extraction is now chunked (hit at unit-800k, fixed in
  `_torch_common.extract_rows`).

## 8. unit-800k, FISTA CV and A100 rows

Harness rows (H200, tol 1e-4 where applicable; the unit-800k tier as served to these runs had T=64):

| tier | solver | task | time s | result |
|---|---|---|---|---|
| unit-100k | torch_fista | CV | 1043.5 | R2 reliable 0.3257, nnz median 120 (same as exact; torch_ws: 33 s) |
| unit-800k | torch_prox_sgd (300 ep) | P | 82.8 | gap median 0.23, nnz median 1.7e5 |
| unit-800k | torch_prox_adam (300 ep, lr 1e-2) | P | 115.6 | diverged: gap median 38.5, nnz median 5.8e5 |
| unit-800k | torch_prox_adam_mb (bs 128, 30 ep, lr 1e-2) | P | 68.4 | diverged: gap median 201 |
| unit-800k | torch_fista screen | P | timeout (3000) | recorded as `status=timeout`; torch_ws: 48.7 s |

- The first unit-800k gradient-optimizer rows errored in sparse extraction (`torch.nonzero` refuses tensors with
  more than 2^31 elements; a wave of ~9000 slots x 780k features). Fixed in `_torch_common.extract_rows` (chunked);
  the rows above are the reruns.
- The remaining prox-Adam failure on unit-800k is not a crash but the learning rate: lr 1e-2 that worked on
  unit-100k diverges at p = 780k (Adam's per-coordinate step does not shrink with p, unlike 1/L). Prox-SGD with
  step 1/L is stable on both tiers. Not tuned further (the round-2 decision below makes it moot).
- The A100 FISTA harness job (63236137) was cancelled before it started: after the round-1 decision (full-p FISTA
  dropped for the unit basis) it would only spend ~10 GPU-hours confirming a known timeout. A100 numbers for
  FISTA exist on synthetic data (section 3) and pca1k-scale problems. The rest of the H200 FISTA job (unit-800k
  P without screening and CV) was cancelled after the screened P timed out, for the same reason.

## 9. Keep / drop

| idea | verdict | evidence |
|---|---|---|
| batched FISTA over (target, alpha) with full X | keep for small p / PCA tiers only; drop for unit tiers | sec 3, 5 |
| warm-start path vs all alphas at once | warm start (segments 1-5); all-at-once 1.7x slower | sec 4 |
| per-problem gap stopping + slot refill/compaction | keep (needed for any batched solver) | - |
| fused prox Triton kernel | keep (25%) | sec 4 |
| gradient restart | keep (10%) | sec 4 |
| batched CV via row masks on a shared X | keep; port to other solvers (CV cost ~ one path) | syn-s, pca1k CV |
| TF32 / bf16 matmuls | drop for exact solvers (cannot reach 1e-4); fine for approximate | sec 4 |
| float64 | use for tol < 1e-5 (H200: 1.4x per iteration) | sec 4 |
| union strong-rule screening in FISTA | drop (keeps 59-79% of p, more churn than gain) | sec 4, 5 |
| CUDA graphs | drop (GEMM-bound) | sec 4 |
| prox-Adam / ISTA / subgradient as Lasso solvers | drop as solvers; keep the R2 observation | sec 6 |
| Hadamard u*v Adam + final prox | keep as the DL-optimizer reference (matches exact CV) | sec 6 |
| minibatch / host-streamed X | keep as the out-of-memory fallback (about 2x cost) | sec 6 |

## 10. Requests to the coordinator

- No contract change needed. The `fit_cv(X, Y, alphas, folds)` protocol in `runner.py` worked as is.
- Suggest the harness also store per-problem iteration counts from `solver_info` when present (FISTA reports
  `iters`, `rel_gap` arrays; currently dropped as non-scalar).
- Modelling note for spec section 8: loose (early-stopped) proximal solutions beat exact Lasso and Ridge on
  unit-100k reliable channels; elastic-net / relaxed-Lasso deserve a row.

## 11. Round 2: do gradient methods keep their R2 advantage at matched sparsity? (unit-100k)

Setup (`scripts/dev_prox_sparsity.py`, H200, results `STORE/results/prox/sparsity_unit-100k.json`, table via
`scripts/dev_prox_sparsity_table.py`): the 822 training rows are split 658 fit / 164 validation (seed 0); every
choice (alpha, ridge lambda for refits, IHT stopping iteration) is made per channel on validation; the 206 test
rows are only used for the numbers below. Mean test R2 over the 20 reliable channels. Ridge on all features
(lambda on validation) gives 0.335 on this split.

Methods: exact Lasso path (torch_ws, tol 1e-4; `lasso` = standard grid to 1e-2 alpha_max, `lasso_ext` = 70 alphas
to 1e-3 alpha_max so the path can reach large k), prox-SGD (ISTA 1000 ep), prox-Adam (300 ep, lr 1e-2), Hadamard
u*v Adam (1000 ep + final prox), and two hard-sparsity trainers: `iht` (accelerated projected gradient, step 1/L,
top-k projection every step) and `iht_adam` (Adam step + top-k projection), stopped at 100/300/1000 iterations.
Readouts: `path` = Lasso solution whose nnz is closest to k; `trunc` = top-k |w| (alpha on validation); `refit` =
top-k support then ridge refit on the kept units ((alpha, lambda) on validation); `path_refit` = relaxed Lasso.

Mean test R2 (paired difference vs relaxed exact Lasso `lasso_ext path_refit` +- paired SE over the 20 channels):

| method | readout | k=10 | k=30 | k=100 | k=300 | k=1000 |
|---|---|---|---|---|---|---|
| exact Lasso | path | 0.114 | 0.213 | 0.312 | 0.305 | 0.172 |
| **exact Lasso** | **path + ridge refit (relaxed)** | **0.214** | **0.280** | **0.332** | **0.324** | **0.305** |
| exact Lasso | top-k + refit | 0.212 (-0.003+-0.008) | 0.252 (-0.028+-0.008) | 0.326 (-0.006+-0.005) | 0.320 (-0.003+-0.005) | 0.311 (+0.006+-0.008) |
| prox-SGD | top-k | 0.071 (-0.144+-0.017) | 0.134 (-0.147+-0.020) | 0.232 (-0.100+-0.013) | 0.310 (-0.014+-0.007) | 0.351 (+0.046+-0.006) |
| prox-SGD | top-k + refit | 0.204 (-0.011+-0.011) | 0.251 (-0.029+-0.008) | 0.309 (-0.023+-0.008) | 0.332 (+0.008+-0.011) | 0.333 (+0.028+-0.011) |
| prox-Adam | top-k | 0.168 (-0.047+-0.012) | 0.240 (-0.040+-0.011) | 0.289 (-0.043+-0.009) | 0.334 (+0.010+-0.005) | 0.350 (+0.045+-0.006) |
| prox-Adam | top-k + refit | 0.199 (-0.015+-0.012) | 0.267 (-0.013+-0.007) | 0.315 (-0.017+-0.011) | 0.333 (+0.009+-0.005) | 0.335 (+0.030+-0.010) |
| Hadamard u*v | top-k | 0.163 (-0.051+-0.014) | 0.243 (-0.038+-0.011) | 0.312 (-0.020+-0.004) | 0.308 (-0.016+-0.005) | 0.307 (+0.002+-0.007) |
| Hadamard u*v | top-k + refit | 0.218 (+0.003+-0.008) | 0.254 (-0.026+-0.008) | 0.324 (-0.008+-0.007) | 0.320 (-0.004+-0.005) | 0.314 (+0.009+-0.009) |
| IHT (hard top-k) | as trained | 0.185 (-0.029+-0.013) | 0.209 (-0.071+-0.014) | 0.276 (-0.056+-0.015) | 0.315 (-0.009+-0.013) | 0.318 (+0.013+-0.012) |
| IHT (hard top-k) | + refit | 0.209 (-0.006+-0.008) | 0.249 (-0.031+-0.010) | 0.285 (-0.047+-0.009) | 0.323 (-0.000+-0.009) | 0.340 (+0.035+-0.008) |
| IHT-Adam | + refit | 0.126 (-0.089+-0.016) | 0.162 (-0.118+-0.025) | 0.208 (-0.124+-0.013) | 0.232 (-0.092+-0.013) | 0.259 (-0.046+-0.009) |

(IHT-Adam without refit is unstable: test R2 down to -0.83.) Fit times on the 658-row split: Lasso 41.7 s
(extended grid 84.1 s), prox-SGD 28.8 s, prox-Adam 12.4 s, Hadamard 48.5 s, IHT 4.2 s for all five k at once.

Caveat at k=1000: the Lasso path does not reach 1000 nonzeros on either grid (median nnz of the densest point: 614
standard, 679 extended), so the k=1000 column compares ~650-unit Lasso supports with 1000-unit gradient supports;
it is the dense, ridge-like regime (prox-SGD/Adam top-1000 reach 0.35, above ridge's 0.335 on this split).

**Answer: no.** At the sparse operating points the readout is for (k <= 100), no gradient-based method beats the
relaxed exact Lasso at the same k: every difference is <= +0.004 and most are significantly negative (prox-SGD /
prox-Adam truncations lose 0.04-0.15; hard-sparsity IHT loses 0.03-0.07 before refit). Hadamard u*v tracks exact
Lasso (it converges to the same solutions). The only positive differences are at k = 300 (+0.008 to +0.010,
1-2 SE, i.e. not established with 20 channels) and at k = 1000, where exact Lasso cannot reach k at all. The early-
stopped R2 advantage reported in section 6 is a density effect, not a better sparse readout. Recommendation: use
exact Lasso (torch_ws) plus a ridge refit on its support (relaxed Lasso adds +0.02 to +0.10 over the plain path at
k <= 30 and +0.02 at k = 100-300); stop investing in gradient-based sparse methods.
