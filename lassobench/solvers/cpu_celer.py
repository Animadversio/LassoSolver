"""celer (working sets + dual extrapolation) Lasso path per target (owned by bench-harness).

celer stops when its gap <= tol * ||y||^2 / n, so tol/2 matches the contract (celer's gap uses an
extrapolated dual point, tighter than the harness's rescaled-residual gap). celer holds the GIL, so
targets are spread over processes (backend=loky, default); each worker copies X once.
"""
from __future__ import annotations

import time
from functools import partial

import numpy as np
import scipy.sparse as sp

from lassobench.api import PathResult, Solver
from lassobench.solvers.cpu_common import assemble, per_target_parallel, per_target_processes


def _celer_one(X, Y, alphas, t, tol, max_iter, max_epochs, prune):
    from celer import celer_path
    y = np.ascontiguousarray(Y[:, t])
    res = celer_path(X, y, "lasso", alphas=alphas[:, t], tol=tol, max_iter=max_iter,
                     max_epochs=max_epochs, prune=prune, return_n_iter=True)
    return sp.csc_matrix(res[1].astype(np.float32)), np.asarray(res[2]), np.asarray(res[-1])


def _factory(X, Y, alphas, **kw):
    return partial(_celer_one, X, Y, alphas, **kw)


class Celer(Solver):
    name = "cpu_celer"
    device = "cpu"

    def fit_path(self, X, Y, alphas):
        dtype = np.dtype(self.kw.get("dtype", "float64"))
        t0 = time.perf_counter()
        Xf = np.asfortranarray(X, dtype=dtype)
        Y = np.ascontiguousarray(Y, dtype)
        t_prep = time.perf_counter() - t0
        n, p = Xf.shape
        A, T = alphas.shape
        kw = dict(tol=self.tol / 2, max_iter=self.max_iter, max_epochs=int(self.kw.get("max_epochs", 50000)),
                  prune=int(self.kw.get("prune", 1)))
        if self.kw.get("backend", "loky") == "threading":
            out = per_target_parallel(lambda t: _celer_one(Xf, Y, alphas, t, **kw), T, self.kw.get("n_jobs"))
        else:
            out = per_target_processes(partial(_factory, **kw), Xf, Y, alphas, self.kw.get("n_jobs"))
        coefs = assemble([o[0] for o in out], p, A)
        iters = np.stack([o[2] for o in out], 1)
        return PathResult(coefs, dict(time_prepare_s=t_prep, n_iter_max=int(iters.max()),
                                      n_iter_total=int(iters.sum()), dtype=str(dtype)))


SOLVERS = {"cpu_celer": Celer}
