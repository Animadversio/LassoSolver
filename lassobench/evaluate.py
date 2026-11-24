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


def edge_fractions(alphas, best_idx):
    """Fraction of targets whose chosen alpha is the smallest / largest alpha of its grid column."""
    al = np.asarray(alphas, np.float64)
    bi = np.asarray(best_idx)
    if al.ndim != 2 or al.shape[0] < 2:
        return {}
    cols = np.arange(len(bi)) % al.shape[1]
    chosen = al[bi, cols]
    return dict(frac_best_smallest_alpha=float(np.mean(np.isclose(chosen, al.min(0)[cols]))),
                frac_best_largest_alpha=float(np.mean(np.isclose(chosen, al.max(0)[cols]))))


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
        s.update(edge_fractions(alphas, bi))
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


# ----------------------------------------------------------------------------- round-2 analysis (spec section 9)

BUDGETS = (10, 30, 100, 300, 1000)
TOPCORR_FRAC = 0.01  # "top-|corr| units" = top 1% of features by |corr(x_j, y_t)| on the training rows


def _support(W, t):
    W = W if sp.isspmatrix_csc(W) else sp.csc_matrix(W)
    lo, hi = W.indptr[t], W.indptr[t + 1]
    return set(W.indices[lo:hi][W.data[lo:hi] != 0].tolist())


def _jac(a, b):
    u = len(a | b)
    return 1.0 if u == 0 else len(a & b) / u


def select_alphas(fold_mse):
    """fold_mse (K, A, T) -> (best_min, best_1se, mean, se). Index 0 = largest alpha (sparsest)."""
    K = fold_mse.shape[0]
    mean = fold_mse.mean(0)
    se = fold_mse.std(0, ddof=1) / np.sqrt(K)
    bmin = np.argmin(mean, 0)
    T = mean.shape[1]
    thr = mean[bmin, np.arange(T)] + se[bmin, np.arange(T)]
    b1 = np.array([int(np.flatnonzero(mean[:, t] <= thr[t])[0]) for t in range(T)])
    return bmin, b1, mean, se


def _ridge_gcv_fit(Xs, y):
    from sklearn.linear_model import RidgeCV
    if Xs.shape[1] == 0:
        return np.zeros(0), 0.0
    s = float((Xs * Xs).sum() / Xs.shape[0]) or 1.0
    m = RidgeCV(alphas=s * np.logspace(-6, 4, 31), fit_intercept=True).fit(Xs, y)
    return m.coef_, float(m.intercept_)


def relaxed_refit(Xtr, Ytr, Xte, Yte, supports, mode="ridge"):
    """Per-channel refit on its support. mode 'ols' (min-norm lstsq) or 'ridge' (GCV). Returns test R2 (T,)."""
    T = Ytr.shape[1]
    pred = np.zeros((Xte.shape[0], T))
    for t in range(T):
        S = np.array(sorted(supports[t]), dtype=np.int64)
        if S.size == 0:
            continue
        Xs = np.asarray(Xtr[:, S], np.float64)
        Xst = np.asarray(Xte[:, S], np.float64)
        y = np.asarray(Ytr[:, t], np.float64)
        if mode == "ols":
            mu, ym = Xs.mean(0), y.mean()
            w = np.linalg.lstsq(Xs - mu, y - ym, rcond=None)[0]
            pred[:, t] = (Xst - mu) @ w + ym
        else:
            w, b = _ridge_gcv_fit(Xs, y)
            pred[:, t] = Xst @ w + b
    return r2_per_channel(Yte, pred)


def topk_supports(W, k):
    W = np.asarray(W.toarray() if sp.issparse(W) else W)
    out = []
    for t in range(W.shape[1]):
        a = np.abs(W[:, t])
        nz = np.flatnonzero(a)
        kk = min(k, nz.size)
        out.append(set(nz[np.argpartition(-a[nz], kk - 1)[:kk]].tolist()) if kk else set())
    return out


def _truncate(W, supports):
    W = np.asarray(W.toarray() if sp.issparse(W) else W, np.float64)
    M = np.zeros_like(W)
    for t, S in enumerate(supports):
        S = np.array(sorted(S), dtype=np.int64)
        M[S, t] = W[S, t]
    return M


def abs_corr(X, Y, chunk=65536):
    n, p = X.shape
    Yc = np.asarray(Y, np.float64)
    Yc = (Yc - Yc.mean(0)) / Yc.std(0)
    C = np.empty((p, Y.shape[1]), np.float32)
    for j in range(0, p, chunk):
        Xc = np.asarray(X[:, j:j + chunk], np.float64)
        Xc = (Xc - Xc.mean(0)) / np.maximum(Xc.std(0), 1e-12)
        C[j:j + chunk] = np.abs(Xc.T @ Yc / n)
    return C


def topcorr_frac(X, Y, supports, frac=TOPCORR_FRAC, chunk=65536, C=None):
    """Fraction of each channel's selected units that rank in the top `frac` of |corr(x_j, y_t)| (train rows)."""
    n, p = X.shape
    if C is not None:
        return _topcorr_from(C, supports, frac)
    Yc = np.asarray(Y, np.float64)
    Yc = (Yc - Yc.mean(0)) / Yc.std(0)
    C = np.empty((p, Y.shape[1]), np.float32)
    for j in range(0, p, chunk):
        Xc = np.asarray(X[:, j:j + chunk], np.float64)
        Xc = (Xc - Xc.mean(0)) / np.maximum(Xc.std(0), 1e-12)
        C[j:j + chunk] = np.abs(Xc.T @ Yc / n)
    return _topcorr_from(C, supports, frac)


def _topcorr_from(C, supports, frac):
    p = C.shape[0]
    m = max(1, int(round(frac * p)))
    out = np.full(C.shape[1], np.nan)
    for t, S in enumerate(supports):
        if S:
            thr = np.partition(C[:, t], p - m)[p - m]
            out[t] = np.mean(C[np.array(sorted(S)), t] >= thr)
    return out


def _rel_stats(v, rel):
    v = np.asarray(v, np.float64)
    vv = v[rel] if rel.any() else v
    return dict(mean=float(np.nanmean(vv)), median=float(np.nanmedian(vv)))


def cv_analysis(tier, alphas, fold_paths, path, folds, half_paths=None, ridge=None):
    """Round-2 metrics from per-fold paths and the full-data path (lists of A sparse (p, T)).

    fold_paths: K lists; path: refit on all training rows; half_paths: two paths on disjoint training halves.
    ridge: dict(coef (p,T) dense, r2 (T,)) of the RidgeCV reference on this tier.
    Returns (summary dict, per-channel arrays dict). All "reliable" stats are over tier.reliable_mask().
    """
    X, Y, Xt, Yt = tier.X, np.asarray(tier.Y), tier.X_test, np.asarray(tier.Y_test)
    rel = tier.reliable_mask()
    A, T = alphas.shape
    K = len(folds)
    fold_mse = np.zeros((K, A, T))
    for k, (tr, va) in enumerate(folds):
        Xva, Yva = np.asarray(X[va]), Y[va].astype(np.float64)
        for a in range(A):
            fold_mse[k, a] = ((Yva - predict(Xva, fold_paths[k][a])) ** 2).mean(0)
    bmin, b1, cvmean, se = select_alphas(fold_mse)
    r2_path = np.stack([r2_per_channel(Yt, predict(Xt, path[a])) for a in range(A)])      # (A, T)
    nnz_path = np.stack([np.diff(sp.csc_matrix(path[a]).indptr) for a in range(A)])       # (A, T)
    S, per = {}, dict(r2_path=r2_path, nnz_path=nnz_path, cv_mse_mean=cvmean, cv_mse_se=se,
                      best_min=bmin, best_1se=b1)
    if ridge is not None:
        S["ridge_r2_reliable"] = _rel_stats(ridge["r2"], rel)["mean"]
    cols = np.arange(T)
    Cabs = abs_corr(X, Y)
    for rule, b in (("min", bmin), ("1se", b1)):
        sup = [_support(path[b[t]], t) for t in range(T)]
        r2 = r2_path[b, cols]
        d = dict(r2_reliable=_rel_stats(r2, rel), nnz_reliable=_rel_stats([len(s) for s in sup], rel),
                 nnz_all_median=float(np.median([len(s) for s in sup])))
        # stability (a): across the K fold fits at this channel's selected alpha (mean pairwise Jaccard)
        jf = np.array([np.mean([_jac(_support(fold_paths[i][b[t]], t), _support(fold_paths[j][b[t]], t))
                                for i in range(K) for j in range(i + 1, K)]) for t in range(T)])
        d["jaccard_folds_reliable"] = _rel_stats(jf, rel)
        per[f"jaccard_folds_{rule}"] = jf
        if half_paths is not None:
            jh = np.array([_jac(_support(half_paths[0][b[t]], t), _support(half_paths[1][b[t]], t)) for t in range(T)])
            d["jaccard_halves_reliable"] = _rel_stats(jh, rel)
            per[f"jaccard_halves_{rule}"] = jh
        tc = topcorr_frac(X, Y, sup, C=Cabs)
        d["topcorr_frac_reliable"] = _rel_stats(tc, rel)
        for mode in ("ols", "ridge"):   # naive: penalty / support chosen on the same rows -> optimistic-biased fit
            rr = relaxed_refit(X, Y, Xt, Yt, sup, mode)
            d[f"relaxed_{mode}_naive_r2_reliable"] = _rel_stats(rr, rel)
            per[f"relaxed_{mode}_naive_{rule}"] = rr
        # honest relaxed Lasso: refit penalty chosen by CV with supports re-selected inside each fold
        fsup = [[_support(fold_paths[k][b[t]], t) for t in range(T)] for k in range(K)]
        rr, lam = relaxed_refit_cv(X, Y, Xt, Yt, folds, fsup, sup)
        d["relaxed_cv_r2_reliable"] = _rel_stats(rr, rel)
        d["relaxed_cv_penalty_median"] = float(np.median(lam))
        per[f"relaxed_cv_{rule}"] = rr
        per[f"r2_{rule}"] = r2
        S[rule] = d
    # frontier along the path: per channel and budget k, the CV-best path point with full-data nnz <= k
    fr = {}
    for k in BUDGETS:
        ok = nnz_path <= k
        idx = np.array([np.flatnonzero(ok[:, t])[np.argmin(cvmean[ok[:, t], t])] for t in range(T)])
        fr[str(k)] = dict(r2_reliable=_rel_stats(r2_path[idx, cols], rel)["mean"],
                          nnz_reliable_mean=_rel_stats(nnz_path[idx, cols], rel)["mean"])
        per[f"frontier_idx_k{k}"] = idx
    S["frontier_path"] = fr
    if ridge is not None:
        S["frontier_topk_ridge"] = topk_frontier(X, Y, Xt, Yt, ridge["coef"], rel, folds, ridge.get("fold_coefs"))
    return S, per


def topk_frontier(X, Y, Xt, Yt, W, rel, folds=None, fold_coefs=None, budgets=BUDGETS):
    """Top-k |w| truncation of dense coefficients W: R2 without refit, and with a CV-honest ridge refit
    (supports re-selected from each fold's own coefficients) when fold_coefs are given."""
    out = {}
    for k in budgets:
        sup = topk_supports(W, k)
        trunc = r2_per_channel(Yt, predict(Xt, _truncate(W, sup)))
        d = dict(trunc_r2_reliable=_rel_stats(trunc, rel)["mean"])
        if fold_coefs is not None:
            fsup = [topk_supports(Wk, k) for Wk in fold_coefs]
            rr, _ = relaxed_refit_cv(X, Y, Xt, Yt, folds, fsup, sup)
            d["refit_cv_r2_reliable"] = _rel_stats(rr, rel)["mean"]
        out[str(k)] = d
    return out


def ridge_halves_stability(X, Y, rel, halves, budgets=(30, 100)):
    """Top-k support Jaccard of RidgeCV fits on two disjoint halves (dense-method stability reference)."""
    from sklearn.linear_model import RidgeCV
    Ws = []
    for h in halves:
        Xh = np.asarray(X[h], np.float64)
        s = float((Xh * Xh).sum() / Xh.shape[0])
        m = RidgeCV(alphas=s * np.logspace(-4, 8, 49), alpha_per_target=True, fit_intercept=True).fit(Xh, np.asarray(Y[h], np.float64))
        Ws.append(m.coef_.T)
    out = {}
    for k in budgets:
        a, b = topk_supports(Ws[0], k), topk_supports(Ws[1], k)
        out[str(k)] = _rel_stats([_jac(a[t], b[t]) for t in range(len(a))], rel)
    return out


REFIT_GRID = np.logspace(-3, 4, 15)  # ridge refit penalty, relative to trace(Xs^T Xs)/n of the support


def _svd_ridge_preds(Xs, y, Xv, lams):
    """Ridge (centered, intercept) for several lambdas via one SVD. Returns (len(lams), n_v) predictions."""
    mu, ym = Xs.mean(0), y.mean()
    U, s, Vt = np.linalg.svd(Xs - mu, full_matrices=False)
    Uy = U.T @ (y - ym)
    Xv = Xv - mu
    out = np.empty((len(lams), Xv.shape[0]))
    for i, lam in enumerate(lams):
        w = Vt.T @ (s / (s * s + lam) * Uy)
        out[i] = Xv @ w + ym
    return out


def relaxed_refit_cv(X, Y, Xt, Yt, folds, fold_supports, supports, grid=REFIT_GRID):
    """Relaxed refit with the ridge penalty chosen by CV over the whole select-then-refit procedure.

    fold_supports[k][t]: support selected on fold k's training rows; the refit on those rows is scored on the
    fold's validation rows for every penalty in `grid` (relative to the support's mean squared column norm).
    The best relative penalty is then used for the refit on `supports[t]` with all training rows.
    Returns (test R2 (T,), chosen relative penalty (T,)).
    """
    T = Y.shape[1]
    n = X.shape[0]
    err = np.zeros((len(grid), T))
    for k, (tr, va) in enumerate(folds):
        Xtr, Xva = np.asarray(X[tr]), np.asarray(X[va])
        for t in range(T):
            S = np.array(sorted(fold_supports[k][t]), dtype=np.int64)
            yv = np.asarray(Y[va, t], np.float64)
            if S.size == 0:
                err[:, t] += ((yv - Y[tr, t].mean()) ** 2).sum()
                continue
            Xs = np.asarray(Xtr[:, S], np.float64)
            sc = (Xs * Xs).sum() / Xs.shape[0]
            P = _svd_ridge_preds(Xs, np.asarray(Y[tr, t], np.float64), np.asarray(Xva[:, S], np.float64), grid * sc)
            err[:, t] += ((P - yv[None]) ** 2).sum(1)
    best = grid[np.argmin(err, 0)]
    pred = np.zeros((Xt.shape[0], T))
    for t in range(T):
        S = np.array(sorted(supports[t]), dtype=np.int64)
        y = np.asarray(Y[:, t], np.float64)
        if S.size == 0:
            pred[:, t] = y.mean()
            continue
        Xs = np.asarray(X[:, S], np.float64)
        sc = (Xs * Xs).sum() / n
        pred[:, t] = _svd_ridge_preds(Xs, y, np.asarray(Xt[:, S], np.float64), [best[t] * sc])[0]
    return r2_per_channel(Yt, pred), best
