"""sklearn coordinate-descent Lasso path per target, threads over targets (owned by bench-harness).

sklearn stops when gap_unnormalized <= tol * ||y||^2, i.e. gap/(||y||^2/2n) <= 2*tol, so we pass tol/2
to match the contract (gap <= tol * ||y||^2 / 2n). cd_fast releases the GIL, so threading works and
X is shared without copies.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from lassobench.api import PathResult, Solver
from lassobench.solvers.cpu_common import assemble, per_target_parallel


class SklearnCD(Solver):
    name = "cpu_sklearn_cd"
    device = "cpu"

    def fit_path(self, X, Y, alphas):
        from sklearn.linear_model import lasso_path
        dtype = np.dtype(self.kw.get("dtype", "float64"))
        t0 = time.perf_counter()
        Xf = np.asfortranarray(X, dtype=dtype)
        Y = np.asarray(Y, dtype)
        t_prep = time.perf_counter() - t0
        n, p = Xf.shape
        A, T = alphas.shape

        def one(t):
            y = np.ascontiguousarray(Y[:, t])
            _, coefs, gaps, iters = lasso_path(Xf, y, alphas=alphas[:, t], tol=self.tol / 2, max_iter=self.max_iter,
                                               precompute=False, return_n_iter=True, check_input=False,
                                               copy_X=False)
            # lasso_path sorts alphas decreasingly; ours already are
            return sp.csc_matrix(coefs.astype(np.float32)), np.asarray(gaps), np.asarray(iters)

        out = per_target_parallel(one, T, self.kw.get("n_jobs"))
        coefs = assemble([o[0] for o in out], p, A)
        iters = np.stack([o[2] for o in out], 1)
        return PathResult(coefs, dict(time_prepare_s=t_prep, n_iter_max=int(iters.max()),
                                      n_iter_total=int(iters.sum()), hit_max_iter=int((iters >= self.max_iter).sum()),
                                      dtype=str(dtype)))


SOLVERS = {"cpu_sklearn_cd": SklearnCD}
