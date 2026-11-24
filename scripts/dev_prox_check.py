"""Correctness check of the gpu-prox solvers against sklearn Lasso (fit_intercept=False, tol=1e-8).

usage: python scripts/dev_prox_check.py [--n 300 --p 2000 --T 4 --A 10 --device cpu --solver torch_fista ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lassobench.api import alpha_grid, duality_gap, make_synthetic, objective  # noqa: E402


def sklearn_path(X, Y, alphas):
    from sklearn.linear_model import Lasso
    A, T = alphas.shape
    out = np.zeros((A, X.shape[1], T))
    X64 = X.astype(np.float64)
    for t in range(T):
        m = Lasso(alpha=alphas[0, t], fit_intercept=False, tol=1e-8, max_iter=200000, warm_start=True)
        for a in range(A):
            m.set_params(alpha=alphas[a, t])
            m.fit(X64, Y[:, t].astype(np.float64))
            out[a, :, t] = m.coef_
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--p", type=int, default=2000)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--A", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--solver", default="torch_fista")
    ap.add_argument("--opts", default="{}", help="json kwargs for the solver")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    d = make_synthetic(n=a.n, p=a.p, T=a.T, n_test=50)
    X, Y = d["X"], d["Y"]
    al = alpha_grid(X, Y, n_alphas=a.A, eps=a.eps)
    t0 = time.perf_counter()
    ref = sklearn_path(X, Y, al)
    print(f"sklearn {time.perf_counter() - t0:.2f}s")
    if a.solver == "torch_fista":
        from lassobench.solvers.torch_fista import SOLVERS
    else:
        from lassobench.solvers.torch_proxgrad import SOLVERS
    kw = json.loads(a.opts)
    if a.device:
        kw["device"] = a.device
    kw.setdefault("tol", 1e-8)
    s = SOLVERS[a.solver](**kw)
    t0 = time.perf_counter()
    res = s.fit_path(X, Y, al)
    print(f"{a.solver} {time.perf_counter() - t0:.2f}s  info: total_iters={res.info.get('total_iters')}")
    X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
    worst_rel_obj, worst_gap, worst_coef = 0, 0, 0
    for k in range(a.A):
        Wk = res.coefs[k].toarray().astype(np.float64)
        Pk = objective(X64, Y64, Wk, al[k])
        Pr = objective(X64, Y64, ref[k], al[k])
        gk = duality_gap(X64, Y64, Wk, al[k]) / ((Y64 ** 2).sum(0) / (2 * a.n))
        worst_rel_obj = max(worst_rel_obj, np.max((Pk - Pr) / Pr))
        worst_gap = max(worst_gap, gk.max())
        worst_coef = max(worst_coef, np.abs(Wk - ref[k]).max() / max(1e-12, np.abs(ref[k]).max()))
        print(f"a{k:2d} nnz {int((Wk != 0).sum()):5d}/{int((ref[k] != 0).sum()):5d}  relobj {np.max((Pk - Pr) / Pr):+.2e}"
              f"  relgap {gk.max():.2e}  iters {res.info['iters'][k] if 'iters' in res.info else '-'}")
    print(f"WORST rel obj {worst_rel_obj:.3e}  rel gap {worst_gap:.3e}  coef rel err {worst_coef:.3e}")


if __name__ == "__main__":
    main()
