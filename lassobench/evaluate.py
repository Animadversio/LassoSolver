"""Metrics for the Lasso benchmark (owned by bench-harness).

All quality numbers are computed by the harness in float64 from the returned sparse coefficients,
never taken from solver self-reports. Formulas match `lassobench.api.objective / duality_gap`, batched over
alphas and chunked over features.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

LOOSE, TIGHT = 1e-4, 1e-6


def _csc(W, p, T):
    if sp.issparse(W):
        return sp.csc_matrix(W)
    W = np.asarray(W)
    assert W.shape == (p, T), W.shape
    return sp.csc_matrix(W)


def predict(X, W, dtype=np.float64):
    """X @ W for sparse/dense W (p, T), touching only the support rows of X."""
    if not sp.issparse(W):
        W = np.asarray(W)
        return np.asarray(X, dtype) @ W.astype(dtype)
    W = sp.csr_matrix(W)
    rows = np.flatnonzero(np.diff(W.indptr))
    if rows.size == 0:
        return np.zeros((X.shape[0], W.shape[1]), dtype)
    return np.asarray(X[:, rows], dtype) @ W[rows].toarray().astype(dtype)


def max_abs_corr(X, R, chunk=8192, dtype=np.float64):
    """max_j |x_j^T r_c| for every column c of R. X (n, p) any array-like, R (n, C)."""
    out = np.zeros(R.shape[1], np.float64)
    R = np.asarray(R, dtype)
    for j in range(0, X.shape[1], chunk):
        Xc = np.asarray(X[:, j:j + chunk], dtype)
        np.maximum(out, np.abs(Xc.T @ R).max(axis=0), out=out)
    return out


def path_quality(X, Y, coefs, alphas, ref=None, alpha_rows=None, chunk=8192):
    """Objective, duality gap and suboptimality for every (alpha, target).

    coefs: list of A sparse (p, T). alphas: (A, T). ref: dict with 'objective' (A, T) or None.
    alpha_rows: optional subset of alpha indices to evaluate (for huge tiers).
    Returns dict of (A', T) arrays: objective, gap_rel, subopt_rel (if ref), nnz, plus alpha_rows.
    """
    n, p = X.shape
    T = Y.shape[1]
    A = len(coefs)
    rows = np.arange(A) if alpha_rows is None else np.asarray(alpha_rows)
    Y64 = np.asarray(Y, np.float64)
    ynorm = (Y64 ** 2).sum(0) / (2 * n)                       # ||y||^2 / (2n), (T,)
    R = np.empty((n, rows.size * T), np.float64)
    l1 = np.empty((rows.size, T))
    nnz = np.empty((rows.size, T), np.int64)
    for i, a in enumerate(rows):
        W = _csc(coefs[a], p, T)
        R[:, i * T:(i + 1) * T] = Y64 - predict(X, W)
        l1[i] = np.asarray(abs(W).sum(axis=0)).ravel()
        nnz[i] = np.diff(W.indptr)
    corr = max_abs_corr(X, R, chunk).reshape(rows.size, T) / n
    al = np.asarray(alphas, np.float64)[rows]
    R3 = R.reshape(n, rows.size, T)
    rss = (R3 ** 2).sum(0) / (2 * n)                          # (A', T)
    primal = rss + al * l1
    scale = np.minimum(1.0, al / np.maximum(corr, 1e-300))
    # dual = (||y||^2 - ||y - s r||^2) / 2n
    dual = ((Y64 ** 2).sum(0)[None] - ((Y64[:, None, :] - R3 * scale[None]) ** 2).sum(0)) / (2 * n)
    gap = primal - dual
    out = dict(alpha_rows=rows, objective=primal, gap=gap, gap_rel=gap / ynorm[None], nnz=nnz,
               ynorm=ynorm)
    if ref is not None and "objective" in ref:
        P_ref = np.asarray(ref["objective"])[rows]
        out["subopt_rel"] = (primal - P_ref) / P_ref
    return out


def summarize_path(q):
    g = q["gap_rel"]
    s = dict(gap_rel_median=float(np.median(g)), gap_rel_max=float(np.max(g)),
             converged_loose=bool(np.max(g) <= LOOSE), converged_tight=bool(np.max(g) <= TIGHT),
             frac_problems_loose=float(np.mean(g <= LOOSE)), nnz_median=float(np.median(q["nnz"])),
             nnz_max=int(np.max(q["nnz"])), n_alpha_eval=int(len(q["alpha_rows"])))
    if "subopt_rel" in q:
        so = q["subopt_rel"]
        s.update(subopt_rel_median=float(np.median(so)), subopt_rel_max=float(np.max(so)))
    s["converged_level"] = "tight" if s["converged_tight"] else ("loose" if s["converged_loose"] else "none")
    return s


def r2_per_channel(Y_true, Y_pred):
    """1 - SSE/SST per column, SST around the test mean (old repo's compute_D2_per_unit)."""
    Y_true = np.asarray(Y_true, np.float64)
    sst = ((Y_true - Y_true.mean(0)) ** 2).sum(0)
    return 1 - ((Y_true - Y_pred) ** 2).sum(0) / sst


def jaccard_cols(A, B):
    """Per-column support Jaccard between two (p, T) sparse/dense matrices (1.0 if both empty)."""
    A = _csc(A, *A.shape); B = _csc(B, *B.shape)
    out = np.empty(A.shape[1])
    for t in range(A.shape[1]):
        a = set(A.indices[A.indptr[t]:A.indptr[t + 1]][A.data[A.indptr[t]:A.indptr[t + 1]] != 0])
        b = set(B.indices[B.indptr[t]:B.indptr[t + 1]][B.data[B.indptr[t]:B.indptr[t + 1]] != 0])
        u = len(a | b)
        out[t] = 1.0 if u == 0 else len(a & b) / u
    return out


def cv_quality(tier, coef, best_idx, alphas, ref=None):
    """Statistical metrics for Task CV. coef: (p, T) sparse or dense refit at the chosen alphas."""
    Yp = predict(tier.X_test, coef)
    r2 = r2_per_channel(tier.Y_test, Yp)
    rel = tier.reliable_mask()
    if sp.issparse(coef):
        nnz = np.diff(sp.csc_matrix(coef).indptr)
    else:
        nnz = (np.asarray(coef) != 0).sum(0)
    Yp_tr = predict(tier.X, coef)
    out = dict(r2=r2, nnz=nnz, best_idx=np.asarray(best_idx),
               alpha=np.asarray(alphas)[np.asarray(best_idx), np.arange(len(best_idx))]
               if best_idx is not None and np.ndim(alphas) == 2 else np.full(len(r2), np.nan),
               r2_train=r2_per_channel(tier.Y, Yp_tr))
    if ref is not None and best_idx is not None and "coefs" in ref:
        T = coef.shape[1]
        Wref = sp.hstack([ref["coefs"][best_idx[t]][:, t] for t in range(T)]).tocsc()
        out["jaccard_ref"] = jaccard_cols(coef, Wref)
    s = dict(r2_test_mean=float(np.mean(r2)), r2_test_median=float(np.median(r2)),
             r2_train_mean=float(np.mean(out["r2_train"])),
             nnz_mean=float(np.mean(nnz)), nnz_median=float(np.median(nnz)),
             n_reliable=int(rel.sum()))
    if rel.any():
        s.update(r2_test_mean_reliable=float(np.mean(r2[rel])), r2_test_median_reliable=float(np.median(r2[rel])))
    if best_idx is not None:
        bi = np.asarray(best_idx)
        s.update(best_idx_median=float(np.median(bi)), frac_best_at_edge=float(np.mean((bi == 0) | (bi == np.shape(alphas)[0] - 1))))
    if "jaccard_ref" in out:
        s["jaccard_ref_mean"] = float(np.mean(out["jaccard_ref"]))
    return out, s


# ----------------------------------------------------------------------------- reference cache

def ref_dir(tier_name):
    import os
    from lassobench.api import STORE
    return os.path.join(STORE, "cache", "ref", tier_name)


def save_reference(tier_name, coefs, alphas, q, info):
    """coefs stored as one sparse (p, A*T) matrix; q from path_quality."""
    import json, os
    d = ref_dir(tier_name)
    os.makedirs(d, exist_ok=True)
    sp.save_npz(os.path.join(d, "coefs.npz"), sp.hstack([sp.csc_matrix(c) for c in coefs]).tocsc())
    np.savez(os.path.join(d, "ref.npz"), alphas=alphas, objective=q["objective"], gap_rel=q["gap_rel"],
             nnz=q["nnz"])
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(info, f, indent=1, default=str)


def load_reference(tier_name, with_coefs=True):
    import json, os
    d = ref_dir(tier_name)
    if not os.path.exists(os.path.join(d, "ref.npz")):
        return None
    z = np.load(os.path.join(d, "ref.npz"))
    ref = {k: z[k] for k in z.files}
    with open(os.path.join(d, "info.json")) as f:
        ref["info"] = json.load(f)
    if with_coefs:
        C = sp.load_npz(os.path.join(d, "coefs.npz")).tocsc()
        A, T = ref["objective"].shape
        ref["coefs"] = [C[:, a * T:(a + 1) * T] for a in range(A)]
    return ref
