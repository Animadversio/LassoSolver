"""Correctness check of torch_ws / torch_admm vs sklearn Lasso on small synthetic problems.

Usage: python scripts/dev_ws_check.py [solver] [n p T] [tol]
"""
import sys, time
import numpy as np
from sklearn.linear_model import Lasso
from lassobench.api import make_synthetic, alpha_grid, duality_gap, objective
from lassobench.solvers import torch_ws, torch_admm  # noqa

name = sys.argv[1] if len(sys.argv) > 1 else "torch_ws"
n, p, T = (int(v) for v in sys.argv[2:5]) if len(sys.argv) > 4 else (300, 2000, 8)
tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-8
d = make_synthetic(n=n, p=p, T=T, k=20, snr=0.5, seed=1)
X, Y = d["X"], d["Y"]
alphas = alpha_grid(X, Y, n_alphas=50)
print("KW", os.environ.get("KW")) if False else None
Sol = {**torch_ws.SOLVERS, **torch_admm.SOLVERS}[name]
import os
import json
s = Sol(tol=tol, **({"verbose": True} if os.environ.get("V") else {}), **json.loads(os.environ.get("KW", "{}")))
s.fit_path(X[:, :500], Y, alpha_grid(X[:, :500], Y, 5))  # warm-up (kernel compile)
t0 = time.perf_counter(); res = s.fit_path(X, Y, alphas); t1 = time.perf_counter()
print(f"{name} n={n} p={p} T={T} tol={tol}: {t1-t0:.3f}s  timings={ {k: round(v,3) for k,v in res.info['timings'].items()} }")
X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
yn = (Y64 ** 2).sum(0) / (2 * n)
gaps = np.array([duality_gap(X64, Y64, res.coefs[a], alphas[a]) / yn for a in range(len(alphas))])
print(f"rel gap (fp64, api.duality_gap): median {np.median(gaps):.2e} max {gaps.max():.2e}")
# sklearn reference on a few alphas
errs, subs = [], []
for a in [5, 20, 35, 49]:
    for t in range(min(T, 4)):
        m = Lasso(alpha=alphas[a, t], fit_intercept=False, tol=1e-10, max_iter=100000).fit(X64, Y64[:, t])
        w = res.coefs[a][:, t].toarray().ravel()
        P = objective(X64, Y64[:, [t]], w[:, None], [alphas[a, t]])[0]
        Pr = objective(X64, Y64[:, [t]], m.coef_[:, None], [alphas[a, t]])[0]
        subs.append((P - Pr) / Pr)
        errs.append(np.abs(w - m.coef_).max() / max(np.abs(m.coef_).max(), 1e-12))
        if a == 49 and t == 0:
            print(f"  nnz ours {np.count_nonzero(w)} sklearn {np.count_nonzero(m.coef_)}")
print(f"vs sklearn: max rel subopt {max(subs):.2e} (min {min(subs):.2e}), max rel coef err {max(errs):.2e}")
