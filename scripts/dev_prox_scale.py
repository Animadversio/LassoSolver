"""Scaling / ablation runs for the gpu-prox solvers on synthetic data.

Generates make_synthetic-style data (factor-correlated, planted sparse, noisy) directly with torch, runs a
list of solver configs on the same problem and appends one JSON row per config to
STORE/results/prox/<tag>.jsonl. The first config can serve as the reference (--ref 0) for objective
suboptimality and support comparisons.

usage:
  python scripts/dev_prox_scale.py --n 1000 --p 100000 --T 64 --A 50 \
      --configs '[["torch_fista", {"tol":1e-6}], ["torch_fista", {"tol":1e-6, "segments":50}]]'
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lassobench.api import STORE, alpha_grid  # noqa: E402
from lassobench.solvers._torch_common import batched_gap, device_name  # noqa: E402


def gen_synthetic(n, p, T, k=30, snr=1.0, rho=0.5, seed=0, n_test=500, dev="cuda"):
    """Same generative model as api.make_synthetic, built with torch on the GPU (chunked)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    m = n + n_test
    n_fac = max(8, p // 50)
    F = torch.randn(m, n_fac, device=dev, generator=g)
    fac = torch.randint(0, n_fac, (p,), device=dev, generator=g)
    X = torch.empty(m, p, device=dev)
    ch = max(1, int(2e8 // m))
    for j in range(0, p, ch):
        X[:, j:j + ch] = (np.sqrt(rho) * F[:, fac[j:j + ch]]
                          + np.sqrt(1 - rho) * torch.randn(m, min(ch, p - j), device=dev, generator=g))
    mu = X[:n].mean(0); sd = X[:n].std(0, unbiased=False)
    X.sub_(mu).div_(sd)
    W = torch.zeros(p, T, device=dev)
    for t in range(T):
        W[torch.randperm(p, device=dev, generator=g)[:k], t] = torch.randn(k, device=dev, generator=g)
    S = X @ W
    Y = S + torch.randn(S.shape, device=dev, generator=g) * S[:n].std(0) / np.sqrt(snr)
    Y = Y - Y[:n].mean(0)
    out = dict(X=X[:n].cpu().numpy(), Y=Y[:n].cpu().numpy(), X_test=X[n:].cpu().numpy(), Y_test=Y[n:].cpu().numpy())
    del X, F, S, Y
    torch.cuda.empty_cache()
    return out


def get_solver(name, opts):
    from lassobench.solvers import torch_fista, torch_proxgrad
    reg = {**torch_fista.SOLVERS, **torch_proxgrad.SOLVERS}
    return reg[name](**opts)


@torch.no_grad()
def evaluate(X, Y, Xt, Yt, alphas, coefs, dev="cuda"):
    """Independent (solver-free) evaluation on GPU: rel gap (fp32 matmul, fp64 reductions), objective,
    nnz, test R2 per (alpha, target)."""
    n = X.shape[0]
    A, T = alphas.shape
    Xg = torch.as_tensor(X, device=dev); Yg = torch.as_tensor(Y, device=dev).T.contiguous()
    Xtg = torch.as_tensor(Xt, device=dev); Ytg = torch.as_tensor(Yt, device=dev)
    sst = ((Ytg - Ytg.mean(0)) ** 2).sum(0).double()
    ynorm2 = (Yg.double() ** 2).sum(1)
    ninv = torch.full((T,), 1.0 / n, device=dev, dtype=torch.float64)
    rel = np.zeros((A, T)); obj = np.zeros((A, T)); nnz = np.zeros((A, T), np.int64); r2 = np.zeros((A, T))
    for a in range(A):
        Wa = torch.as_tensor(coefs[a].toarray(), device=dev)        # (p, T)
        R = Yg - (Xg @ Wa).T                                          # (T, n)
        C = R @ Xg
        cinf = torch.linalg.vector_norm(C, ord=float("inf"), dim=1).double()
        l1 = Wa.abs().sum(0).double()
        al = torch.as_tensor(alphas[a], device=dev, dtype=torch.float64)
        _, rr = batched_gap(R, cinf, l1, Yg, al, ninv, ynorm2)
        rel[a] = rr.cpu().numpy()
        obj[a] = (0.5 * (R.double() ** 2).sum(1) / n + al * l1).cpu().numpy()
        nnz[a] = (Wa != 0).sum(0).cpu().numpy()
        sse = ((Ytg - Xtg @ Wa) ** 2).sum(0).double()
        r2[a] = (1 - sse / sst).cpu().numpy()
    return dict(rel_gap=rel, obj=obj, nnz=nnz, r2=r2)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parents[1]).decode().strip()
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p", type=int, default=20000)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--A", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--ref", type=int, default=-1, help="index of reference config (-1 none)")
    ap.add_argument("--tag", default="scale")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--warmup", action="store_true", help="run each config once on a tiny problem first")
    a = ap.parse_args()
    configs = json.loads(a.configs)
    t0 = time.perf_counter()
    d = gen_synthetic(a.n, a.p, a.T, snr=a.snr)
    t_gen = time.perf_counter() - t0
    alphas = alpha_grid(d["X"], d["Y"], n_alphas=a.A, eps=a.eps)
    gpu = device_name()
    print(f"# data n={a.n} p={a.p} T={a.T} A={a.A} gen {t_gen:.1f}s gpu={gpu} host={socket.gethostname()}", flush=True)
    outdir = Path(STORE) / "results" / "prox"
    outdir.mkdir(parents=True, exist_ok=True)
    outf = outdir / f"{a.tag}.jsonl"
    if a.warmup:
        dw = gen_synthetic(200, 2000, 4)
        aw = alpha_grid(dw["X"], dw["Y"], n_alphas=5)
        for name, opts in configs:
            get_solver(name, {**opts, "max_iter": 50, "epochs": 5} if "proxgrad" in name or name.startswith("torch_") else opts).fit_path(dw["X"], dw["Y"], aw)
    ref_ev = None
    order = list(range(len(configs)))
    if a.ref >= 0:
        order = [a.ref] + [i for i in order if i != a.ref]
    for i in order:
        name, opts = configs[i]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        s = get_solver(name, opts)
        t0 = time.perf_counter()
        res = s.fit_path(d["X"], d["Y"], alphas)
        t_fit = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        info = res.info
        row = dict(tag=a.tag, solver=name, opts=opts, n=a.n, p=a.p, T=a.T, A=a.A, eps=a.eps, gpu=gpu,
                   host=socket.gethostname(), git=git_hash(), time_wall_s=t_fit, peak_gpu_gb=peak,
                   **{k: v for k, v in info.items() if k.startswith("time_") or k in
                      ("total_iters", "slot_iters", "batch_capacity", "n_restarts", "n_unconverged", "phase_s", "q_mean", "q_max", "n_screen_rebuilds", "n_graph_captures",
                       "lipschitz")})
        if "iters" in info:
            it = np.asarray(info["iters"])
            row.update(iters_med=float(np.median(it)), iters_max=int(it.max()), iters_sum=int(it.sum()),
                       iters_by_alpha=np.median(it, 1).tolist())
        if "history" in info and info["history"] and info["history"][0]:
            row["history"] = info["history"][0]
        if not a.no_eval:
            ev = evaluate(d["X"], d["Y"], d["X_test"], d["Y_test"], alphas, res.coefs)
            row.update(relgap_med=float(np.median(ev["rel_gap"])), relgap_max=float(ev["rel_gap"].max()),
                       relgap_max_by_alpha=ev["rel_gap"].max(1).tolist(),
                       nnz_med=float(np.median(ev["nnz"])), nnz_by_alpha=np.median(ev["nnz"], 1).tolist(),
                       r2_by_alpha=ev["r2"].mean(1).tolist(),
                       r2_oracle=float(ev["r2"].max(0).mean()))
            if ref_ev is None and a.ref >= 0:
                ref_ev = ev
            elif ref_ev is not None:
                subopt = (ev["obj"] - ref_ev["obj"]) / ref_ev["obj"]
                bi = ref_ev["r2"].argmax(0)  # reference's oracle alpha per target
                ti = np.arange(a.T)
                row.update(subopt_med=float(np.median(subopt)), subopt_max=float(subopt.max()),
                           r2_at_ref_alpha=float(ev["r2"][bi, ti].mean()),
                           ref_r2_at_ref_alpha=float(ref_ev["r2"][bi, ti].mean()),
                           nnz_at_ref_alpha=float(np.median(ev["nnz"][bi, ti])),
                           ref_nnz_at_ref_alpha=float(np.median(ref_ev["nnz"][bi, ti])))
        with open(outf, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        keys = ["solver", "time_wall_s", "time_engine_s", "total_iters", "iters_med", "iters_max", "relgap_med",
                "relgap_max", "nnz_med", "r2_oracle", "subopt_max", "r2_at_ref_alpha", "nnz_at_ref_alpha",
                "peak_gpu_gb", "n_unconverged"]
        print(json.dumps({"opts": opts, **{k: row.get(k) for k in keys}}), flush=True)
        if "phase_s" in row:
            print("   phases:", {k: round(v, 3) for k, v in row["phase_s"].items()}, flush=True)
        del res
    print(f"# wrote {outf}")


if __name__ == "__main__":
    main()
