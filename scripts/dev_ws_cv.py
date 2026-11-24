"""Check torch_ws.fit_cv (batched folds) against the harness default_fit_cv (K+1 fit_path calls).
Usage: python scripts/dev_ws_cv.py tier [tol] [max_targets]"""
import sys, time
import numpy as np
from lassobench import runner
from lassobench.api import alpha_grid
from lassobench.data import load_tier
from lassobench.solvers.torch_ws import TorchWS

tier = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
mt = int(sys.argv[3]) if len(sys.argv) > 3 else None
d = load_tier(tier)
X, Y = np.asarray(d.X), np.asarray(d.Y)
if mt: Y = Y[:, :mt]
alphas = alpha_grid(X, Y, 50, 1e-2)
folds = runner.make_folds(X.shape[0])
s = TorchWS(tol=tol)
s.fit_path(X[:, :2000], Y[:, :4], alphas[:5, :4])
t0 = time.perf_counter(); r1 = s.fit_cv(X, Y, alphas, folds); t1 = time.perf_counter()
r2 = runner.default_fit_cv(s, X, Y, alphas, folds); t2 = time.perf_counter()
print(f"{tier} T={Y.shape[1]} tol={tol}: fit_cv batched {t1-t0:.2f}s  default (K+1 fit_path) {t2-t1:.2f}s")
print("best_idx equal:", np.mean(r1["best_idx"] == r2["best_idx"]), " max |cv_mse diff|/mse:",
      np.max(np.abs(r1["cv_mse"] - r2["cv_mse"]) / r2["cv_mse"]))
c1, c2 = r1["coef"].toarray(), r2["coef"].toarray()
print("coef rel diff:", np.abs(c1 - c2).max() / np.abs(c2).max(), " nnz", (c1 != 0).sum(0)[:8], (c2 != 0).sum(0)[:8])
print("fold timings:", [ {k: round(v, 2) for k, v in fi["timings"].items() if v > 0.1} for fi in r1["info"]["fold_infos"]][:1], r1["info"]["cv_batch"])
