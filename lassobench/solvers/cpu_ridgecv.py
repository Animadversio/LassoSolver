"""RidgeCV predictive baseline (owned by bench-harness). Task CV only.

Per-target alpha by efficient leave-one-out (sklearn GCV, eigen mode on the n x n Gram since p >> n).
Ridge alpha grid is relative to the mean Gram eigenvalue: alpha = (trace(XX^T)/n) * logspace(-4, 8, 49)
(for standardized X, trace(XX^T)/n = p).

fit_intercept=True is required even though X and Y are centered on the training rows: centering puts the
all-ones vector in the null space of XX^T, and without the intercept sklearn's LOO-GCV error -> 0 as alpha -> 0,
so every target picked the smallest alpha (test R2 -5.9 on pca1k in the first round). With the intercept
GCV handles that direction; the fitted intercept is ~0 on centered data and is ignored (reported in info).
"""
from __future__ import annotations

import time

import numpy as np

from lassobench.api import Solver


class RidgeCVBaseline(Solver):
    name = "cpu_ridgecv"
    device = "cpu"

    def fit_cv(self, X, Y, alphas, folds):
        from sklearn.linear_model import RidgeCV
        t0 = time.perf_counter()
        X = np.asarray(X, np.float64)
        scale = float((X * X).sum() / X.shape[0])
        grid = scale * np.logspace(float(self.kw.get("lo", -4)), float(self.kw.get("hi", 8)), int(self.kw.get("n", 49)))
        m = RidgeCV(alphas=grid, alpha_per_target=True, fit_intercept=True).fit(X, np.asarray(Y, np.float64))
        best = np.searchsorted(grid, np.atleast_1d(m.alpha_))
        best = np.clip(best, 0, len(grid) - 1)
        coef = np.asarray(m.coef_, np.float32).T  # (p, T)
        A_r = np.repeat(grid[:, None], Y.shape[1], 1)
        return dict(coef=coef, best_idx=best, alphas=A_r, cv_mse=None,
                    info=dict(alpha_scale=scale, time_fit_s=time.perf_counter() - t0, n_ridge_alphas=len(grid),
                              cv="LOO-GCV (ignores harness folds)",
                              intercept_absmax=float(np.max(np.abs(m.intercept_))),
                              frac_alpha_at_edge=float(np.mean((best == 0) | (best == len(grid) - 1)))))


SOLVERS = {"cpu_ridgecv": RidgeCVBaseline}
