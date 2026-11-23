"""Old-repo protocol, for reproducing the Dec-2024 pca1000 timings (owned by bench-harness). Task CV only.

Exactly what `MultiOutputSeparateLassoCV(cv=5, n_alphas=100, n_jobs=-1)` did: sequential loop over targets,
each a sklearn LassoCV with its own 100-alpha grid, its own unshuffled 5 folds, fit_intercept=True, tol=1e-4,
folds parallel over n_jobs. Does NOT use the harness grid/folds, so it is not comparable on solution quality,
only on the protocol cost.
"""
from __future__ import annotations

import time

import numpy as np

from lassobench.api import Solver


class LegacyLassoCV(Solver):
    name = "cpu_sklearn_lassocv_legacy"
    device = "cpu"

    def fit_cv(self, X, Y, alphas, folds):
        from sklearn.linear_model import LassoCV
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        n_jobs = int(self.kw.get("n_jobs", -1))
        coef = np.zeros((X.shape[1], Y.shape[1]), np.float32)
        alpha = np.zeros(Y.shape[1])
        iters = []
        t0 = time.perf_counter()
        for t in range(Y.shape[1]):
            m = LassoCV(cv=5, n_alphas=int(self.kw.get("n_alphas", 100)), n_jobs=n_jobs,
                        max_iter=self.max_iter, tol=self.tol).fit(X, Y[:, t])
            coef[:, t] = m.coef_
            alpha[t] = m.alpha_
            iters.append(int(m.n_iter_))
        return dict(coef=coef, best_idx=None, alphas=alpha[None, :], cv_mse=None,
                    info=dict(alpha=alpha.tolist(), n_iter=iters, time_fit_s=time.perf_counter() - t0,
                              note="intercept fitted; prediction ignores intercept (Y, X centered on train)"))


SOLVERS = {"cpu_sklearn_lassocv_legacy": LegacyLassoCV}
