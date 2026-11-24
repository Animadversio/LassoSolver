"""Elastic-net correctness of torch_ws(l1_ratio=rho) vs sklearn ElasticNet(alpha=alpha/rho, l1_ratio=rho)."""
import sys, time
import numpy as np
from sklearn.linear_model import ElasticNet
from lassobench.api import make_synthetic, alpha_grid
from lassobench.solvers.torch_ws import TorchWS
rho = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
d = make_synthetic(n=300, p=2000, T=6, k=20, snr=0.5, seed=1)
X, Y = d["X"], d["Y"]; n = X.shape[0]
al = alpha_grid(X, Y, 50)
s = TorchWS(tol=1e-8, l1_ratio=rho)
s.fit_path(X[:, :500], Y[:, :2], al[:3, :2])
t0 = time.perf_counter(); res = s.fit_path(X, Y, al); print(f"fit {time.perf_counter()-t0:.2f}s")
X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
def obj(w, y, a):
    r = y - X64 @ w
    return r @ r / (2 * n) + a * np.abs(w).sum() + 0.5 * a * (1 - rho) / rho * w @ w
subs, errs = [], []
for a in [3, 15, 30, 49]:
    for t in range(3):
        m = ElasticNet(alpha=al[a, t] / rho, l1_ratio=rho, fit_intercept=False, tol=1e-12, max_iter=200000).fit(X64, Y64[:, t])
        w = res.coefs[a][:, t].toarray().ravel()
        P, Pr = obj(w, Y64[:, t], al[a, t]), obj(m.coef_, Y64[:, t], al[a, t])
        subs.append((P - Pr) / Pr); errs.append(np.abs(w - m.coef_).max() / np.abs(m.coef_).max())
print(f"rho={rho}: max rel subopt {max(subs):.2e} min {min(subs):.2e} max rel coef err {max(errs):.2e}; nnz last {res.coefs[-1][:, 0].nnz} vs {np.count_nonzero(m.coef_)}")
