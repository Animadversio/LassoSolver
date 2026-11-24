"""Scaling benchmark for torch_ws / torch_admm on synthetic data generated on the GPU.

Same recipe as lassobench.api.make_synthetic (AR-like factor blocks, rho=0.5, k planted, snr), but
generated with torch so p = 1e6, n = 1e4 is feasible. Verifies every alpha's duality gap in float64
on the GPU (independent of the solver's own fp32/fp64 bookkeeping) and appends one JSON line to
STORE/results/ws_dev/scale.jsonl.

Usage: python scripts/dev_ws_scale.py --n 1000 --p 100000 --T 64 --tol 1e-4 [--solver torch_ws] [--kw '{...}']
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from lassobench.api import STORE
from lassobench.solvers import torch_admm, torch_ws

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--p", type=int, default=100000)
ap.add_argument("--T", type=int, default=64)
ap.add_argument("--k", type=int, default=30)
ap.add_argument("--snr", type=float, default=1.0)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--solver", default="torch_ws")
ap.add_argument("--kw", default="{}")
ap.add_argument("--n_alphas", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--celer_targets", type=int, default=0)
ap.add_argument("--save_data", default="")
ap.add_argument("--tag", default="")
args = ap.parse_args()


def make_synth_torch(n, p, T, k, snr, rho=0.5, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_fac = max(8, p // 50)
    F = torch.randn(n, n_fac, device="cuda", generator=g)
    assign = torch.randint(0, n_fac, (p,), device="cuda", generator=g)
    X = torch.empty(n, p, device="cuda")
    ch = max(1, int(2e9 // (4 * n)))
    for j in range(0, p, ch):
        blk = np.sqrt(rho) * F[:, assign[j:j + ch]] + np.sqrt(1 - rho) * torch.randn(n, min(ch, p - j), device="cuda", generator=g)
        blk -= blk.mean(0)
        blk /= blk.std(0, unbiased=False)
        X[:, j:j + ch] = blk
    W = torch.zeros(p, T, device="cuda")
    for t in range(T):
        idx = torch.randperm(p, device="cuda", generator=g)[:k]
        W[idx, t] = torch.randn(k, device="cuda", generator=g)
    S = X @ W
    Y = S + torch.randn(n, T, device="cuda", generator=g) * S.std(0) / np.sqrt(snr)
    Y -= Y.mean(0)
    return X, Y


def verify_gaps(Xd, Yd, coefs, alphas):
    """float64 relative duality gaps (spec definition) for every alpha/target, on GPU."""
    n, p = Xd.shape
    Y64 = Yd.double()
    yn = (Y64 ** 2).sum(0) / (2 * n)
    out = []
    ch = max(1, int(1e9 // (8 * n)))
    for a, W in enumerate(coefs):
        Wc = W.tocoo()
        Wd = torch.zeros(p, W.shape[1], dtype=torch.float64, device="cuda")
        Wd[torch.as_tensor(Wc.row, device="cuda").long(), torch.as_tensor(Wc.col, device="cuda").long()] = torch.as_tensor(Wc.data, device="cuda")
        nzr = torch.nonzero(Wd.abs().sum(1)).ravel()
        R = Y64 - Xd[:, nzr].double() @ Wd[nzr]
        corr = torch.zeros(W.shape[1], dtype=torch.float64, device="cuda")
        for j in range(0, p, ch):
            corr = torch.maximum(corr, (Xd[:, j:j + ch].double().T @ R).abs().max(0).values)
        corr /= n
        al = torch.as_tensor(alphas[a], device="cuda", dtype=torch.float64)
        sc = torch.clamp(al / torch.clamp(corr, min=1e-30), max=1.0)
        th = R * sc
        primal = (R * R).sum(0) / (2 * n) + al * Wd.abs().sum(0)
        dual = ((Y64 * Y64).sum(0) - ((Y64 - th) ** 2).sum(0)) / (2 * n)
        out.append(((primal - dual) / yn).cpu().numpy())
    return np.array(out)


torch.manual_seed(args.seed)
t0 = time.perf_counter()
Xd, Yd = make_synth_torch(args.n, args.p, args.T, args.k, args.snr, seed=args.seed)
torch.cuda.synchronize()
t_gen = time.perf_counter() - t0
X = Xd.cpu().numpy()
Y = Yd.cpu().numpy()
if args.save_data:
    os.makedirs(args.save_data, exist_ok=True)
    np.save(os.path.join(args.save_data, "X.npy"), X)
    np.save(os.path.join(args.save_data, "Y.npy"), Y)
amax = (Xd.T @ Yd).abs().max(0).values.double().cpu().numpy() / args.n
alphas = amax[None, :] * np.logspace(0, -2, args.n_alphas)[:, None]
del Xd
torch.cuda.empty_cache()

Sol = {**torch_ws.SOLVERS, **torch_admm.SOLVERS}[args.solver]
kw = json.loads(args.kw)
s = Sol(tol=args.tol, **kw)
# warm-up (kernel compile, cuBLAS init) on a small slice
tw = time.perf_counter()
s.fit_path(X[:, :2000], Y[:, :4], alphas[:5, :4] * 0 + alphas[:5, :4])
t_warm = time.perf_counter() - tw
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
res = s.fit_path(X, Y, alphas)
t_fit = time.perf_counter() - t0
torch.cuda.empty_cache()
Xd = torch.from_numpy(X).cuda()
Yd = torch.from_numpy(Y).cuda()
gaps = verify_gaps(Xd, Yd, res.coefs, alphas)
nnz = np.array([[c[:, t].nnz for t in range(args.T)] for c in res.coefs])
info = res.info
pa = info.get("per_alpha", [])
row = dict(solver=args.solver, tag=args.tag, n=args.n, p=args.p, T=args.T, tol=args.tol, kw=kw,
           gpu=torch.cuda.get_device_name(0), time_fit_s=t_fit, time_warmup_s=t_warm,
           gap_median=float(np.median(gaps)), gap_max=float(gaps.max()),
           frac_conv_1e4=float((gaps <= 1e-4).mean()), frac_conv_1e6=float((gaps <= 1e-6).mean()),
           nnz_max=int(nnz.max()), nnz_median_last=float(np.median(nnz[-1])),
           timings={k: round(v, 3) for k, v in info.get("timings", {}).items()},
           peak_gpu_mem_gb=info.get("peak_gpu_mem_gb"), union_final=info.get("union_final"),
           n_full_matmul=info.get("n_full_matmul"),
           ws_max=max([r.get("ws_max", 0) for r in pa], default=None),
           outer_total=sum(r.get("outer", 0) for r in pa) if pa else None,
           iters=info.get("iters"),
           git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
if args.celer_targets:
    from celer import Lasso as CLasso
    X64 = X.astype(np.float64)
    tc = time.perf_counter()
    for t in range(args.celer_targets):
        m = CLasso(alpha=alphas[0, t], fit_intercept=False, tol=args.tol / 2, warm_start=True,
                   max_iter=1000, max_epochs=100000)
        for a in range(args.n_alphas):
            m.alpha = alphas[a, t]
            m.fit(X64, Y[:, t].astype(np.float64))
    row["celer_s_per_target"] = (time.perf_counter() - tc) / args.celer_targets
    row["celer_extrap_T_s"] = row["celer_s_per_target"] * args.T
print(json.dumps(row), flush=True)
os.makedirs(f"{STORE}/results/ws_dev", exist_ok=True)
with open(f"{STORE}/results/ws_dev/scale.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
