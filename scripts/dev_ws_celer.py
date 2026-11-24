"""celer yardstick on the same synthetic problems (CPU). Usage: dev_ws_celer.py n p T tol [n_targets_to_time]"""
import sys, time
import numpy as np
from celer import Lasso as CLasso
from lassobench.api import make_synthetic, alpha_grid, duality_gap
n, p, T = (int(v) for v in sys.argv[1:4]); tol = float(sys.argv[4]); Tt = int(sys.argv[5]) if len(sys.argv) > 5 else T
d = make_synthetic(n=n, p=p, T=T, k=20, snr=0.5, seed=1)
X, Y = d["X"].astype(np.float64), d["Y"].astype(np.float64)
alphas = alpha_grid(X, Y, n_alphas=50)
yn = (Y ** 2).sum(0) / (2 * n)
t0 = time.perf_counter(); gmax = 0; nnz = []
for t in range(Tt):
    m = CLasso(alpha=alphas[0, t], fit_intercept=False, tol=tol / 2, warm_start=True, max_iter=1000, max_epochs=100000)
    for a in range(len(alphas)):
        m.alpha = alphas[a, t]; m.fit(X, Y[:, t])
    nnz.append(np.count_nonzero(m.coef_))
t1 = time.perf_counter()
print(f"celer n={n} p={p}: {t1-t0:.2f}s for {Tt} targets x 50 alphas ({(t1-t0)/Tt:.3f}s/target); nnz at alpha_min: {nnz[:8]}")
