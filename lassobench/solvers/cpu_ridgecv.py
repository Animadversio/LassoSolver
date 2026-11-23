"""RidgeCV predictive baseline (owned by bench-harness). Task CV only.

Per-target alpha by efficient leave-one-out (sklearn GCV, eigen mode on the n x n Gram since p >> n).
Ridge alpha grid is relative to the mean Gram eigenvalue: alpha = (trace(XX^T)/n) * logspace(-6, 4, 41)
(for standardized X, trace(XX^T)/n = p).
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
        grid = scale * np.logspace(float(self.kw.get("lo", -6)), float(self.kw.get("hi", 4)), int(self.kw.get("n", 41)))
        m = RidgeCV(alphas=grid, alpha_per_target=True, fit_intercept=False).fit(X, np.asarray(Y, np.float64))
        best = np.searchsorted(grid, np.atleast_1d(m.alpha_))
        best = np.clip(best, 0, len(grid) - 1)
        coef = np.asarray(m.coef_, np.float32).T  # (p, T)
        A_r = np.repeat(grid[:, None], Y.shape[1], 1)
        return dict(coef=coef, best_idx=best, alphas=A_r, cv_mse=None,
                    info=dict(alpha_scale=scale, time_fit_s=time.perf_counter() - t0, n_ridge_alphas=len(grid),
                              cv="LOO-GCV (ignores harness folds)"))


SOLVERS = {"cpu_ridgecv": RidgeCVBaseline}
