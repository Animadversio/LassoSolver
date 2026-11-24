"""Solver contract for the Lasso benchmark. Owned by the coordinator session.

Problem (sklearn convention), solved independently for every target t and every alpha a:

    min_w  1/(2n) * ||y_t - X w||_2^2  +  alpha[a, t] * ||w||_1

X is (n, p) float32, column-standardized (zero mean, unit variance, no constant columns).
Y is (n, T) float32, column-centered. No intercept is fitted. p >> n is the regime of interest.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Storage root for big files (data tiers, caches, raw results). Override with LASSOBENCH_STORE to run from
# another account: point it at a directory you can write to. Tiers are read from `DATA_ROOT`, which defaults to
# STORE/data and can be pointed at the lab's read-only copy with LASSOBENCH_DATA, e.g.
#   export LASSOBENCH_DATA=/n/holylabs/LABS/kempner_alvarez_lab/Lab/shared/LassoSolver/data
#   export LASSOBENCH_STORE=/path/you/own/LassoSolver
STORE = os.environ.get("LASSOBENCH_STORE", "/n/holylabs/LABS/kempner_alvarez_lab/Lab/binxuwang/Projects/LassoSolver")
DATA_ROOT = os.environ.get("LASSOBENCH_DATA", os.path.join(STORE, "data"))


@dataclass
class PathResult:
    """Output of `Solver.fit_path`.

    coefs: list of length A; coefs[a] is a scipy.sparse CSC matrix of shape (p, T) holding the
        weights for alpha row a. Sparse because dense (A, p, T) does not fit at p = 1e6.
    info: free-form solver diagnostics (iterations, final duality gaps, phase timings, ...).
    """
    coefs: list
    info: dict[str, Any] = field(default_factory=dict)


class Solver:
    """Base class. Subclass, set `name`, implement `fit_path`.

    The harness times `fit_path` end to end, *including* any host->GPU transfer and any
    precomputation (Gram matrices, norms, factorizations). X and Y always arrive as numpy
    arrays on the host. A solver may keep state between calls only via its own explicit
    `prepare` cache, whose time is reported separately (see BENCHMARK_SPEC).
    """
    name: str = "base"
    device: str = "cpu"  # "cpu" or "cuda"

    def __init__(self, tol: float = 1e-4, max_iter: int = 10000, **kw):
        self.tol = tol          # target: duality gap <= tol * ||y_t||^2 / (2n), per problem
        self.max_iter = max_iter
        self.kw = kw

    def fit_path(self, X: np.ndarray, Y: np.ndarray, alphas: np.ndarray) -> PathResult:
        """alphas: (A, T), each column decreasing. Return PathResult with A sparse (p, T) coefs."""
        raise NotImplementedError


def alpha_max(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Smallest alpha per target giving the all-zero solution: max_j |x_j^T y_t| / n. Shape (T,)."""
    n = X.shape[0]
    out = np.zeros(Y.shape[1], dtype=np.float64)
    step = max(1, int(2e8 // max(1, X.shape[0])))  # chunk over features to bound memory
    for j in range(0, X.shape[1], step):
        out = np.maximum(out, np.abs(X[:, j:j + step].T @ Y).max(axis=0))
    return out / n


def alpha_grid(X: np.ndarray, Y: np.ndarray, n_alphas: int = 50, eps: float = 1e-2) -> np.ndarray:
    """Per-target log-spaced grid from alpha_max down to eps * alpha_max. Shape (n_alphas, T)."""
    amax = alpha_max(X, Y)
    return amax[None, :] * np.logspace(0, np.log10(eps), n_alphas)[:, None]


def objective(X, Y, W, alphas_row) -> np.ndarray:
    """Primal objective per target for one alpha row. W: (p, T) sparse or dense. Returns (T,)."""
    n = X.shape[0]
    R = Y - (X @ W if not hasattr(W, "toarray") else np.asarray((W.T @ X.T).T))
    l1 = np.asarray(abs(W).sum(axis=0)).ravel()
    return (R * R).sum(axis=0) / (2 * n) + np.asarray(alphas_row) * l1


def duality_gap(X, Y, W, alphas_row) -> np.ndarray:
    """Lasso duality gap per target for one alpha row (dual point by residual rescaling). (T,)."""
    n = X.shape[0]
    a = np.asarray(alphas_row, dtype=np.float64)
    R = Y - (X @ W if not hasattr(W, "toarray") else np.asarray((W.T @ X.T).T))
    corr = alpha_max(X, R)                       # max_j |x_j^T r| / n
    scale = np.minimum(1.0, a / np.maximum(corr, 1e-30))
    theta = R * scale[None, :]                   # dual-feasible residual
    primal = (R * R).sum(axis=0) / (2 * n) + a * np.asarray(abs(W).sum(axis=0)).ravel()
    dual = ((Y * Y).sum(axis=0) - ((Y - theta) ** 2).sum(axis=0)) / (2 * n)
    return primal - dual


def make_synthetic(n=1000, p=20000, T=16, k=30, snr=1.0, rho=0.5, seed=0, n_test=500):
    """Planted-sparse problem with correlated features (AR-like blocks), noisy targets.

    Returns dict with standardized X, centered Y, X_test, Y_test, W_true (p, T).
    Small helper for solver development; real-data tiers come from `lassobench.data`.
    """
    rng = np.random.default_rng(seed)
    m = n + n_test
    n_fac = max(8, p // 50)
    F = rng.standard_normal((m, n_fac)).astype(np.float32)
    load = np.zeros((n_fac, p), dtype=np.float32)
    load[rng.integers(0, n_fac, p), np.arange(p)] = 1.0
    X = np.sqrt(rho) * (F @ load) + np.sqrt(1 - rho) * rng.standard_normal((m, p)).astype(np.float32)
    mu, sd = X[:n].mean(0), X[:n].std(0)
    X = (X - mu) / sd
    W = np.zeros((p, T), dtype=np.float32)
    for t in range(T):
        W[rng.choice(p, k, replace=False), t] = rng.standard_normal(k)
    S = X @ W
    Y = S + rng.standard_normal(S.shape).astype(np.float32) * S[:n].std(0) / np.sqrt(snr)
    ymu = Y[:n].mean(0)
    Y = (Y - ymu).astype(np.float32)
    return dict(X=X[:n], Y=Y[:n], X_test=X[n:], Y_test=Y[n:], W_true=W)
