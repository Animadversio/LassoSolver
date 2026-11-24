#!/usr/bin/env python
"""factorized-readout: unit-basis Lasso side of the comparison (GPU node, torch_ws).

  python scripts/dev_fact_lasso.py --tier unit-100k

1. Harness-protocol Lasso (Task CV, 5-fold, 50 alphas, torch_ws tol 1e-4) via lassobench.runner.run, with its
   row written to a side file (the leaderboard already has this row), keeping the chosen coefficients.
2. Lasso path on all training rows: test R2 and nonzeros per alpha (R2-vs-sparsity frontier).
3. Lasso under the factorized models' 5-member protocol (member m: RandomState(m) 80/20 split of the training
   rows, per-channel alpha by member validation R2, ensemble = prediction average): stability (support Jaccard,
   weight cosine, peak-position distance) on the same footing as the factorized readouts.
4. Spatial concentration of every Lasso solution: fraction of sum_c |w| at the best single position and in the
   best 2 x 2 window, against a null that keeps each selected unit's channel and weight but moves it to a random
   position.
Saves STORE/results/factorized/<tier>/lasso.npz and lasso_summary.json; appends one `fact_lasso_members` row.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lassobench import api, runner  # noqa: E402
from lassobench.factorized.data import LAYER_MAPS, OUT_DIR, member_split  # noqa: E402
from lassobench.data import load_tier  # noqa: E402


def r2(Yt, P):
    Yt = np.asarray(Yt, np.float64)
    return 1 - ((Yt - P) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)


def full_w(W, keep, p):
    """(p_kept, T) sparse/dense -> dense (p, T) in the full (C, H, W) unit order."""
    W = W.toarray() if sp.issparse(W) else np.asarray(W)
    out = np.zeros((p, W.shape[1]), np.float32)
    out[keep] = W
    return out


def concentration(Wfull, shape, n_null=200, seed=0):
    """Per channel: frac of sum|w| at best position / best 2x2 window, peak (row, col), and null means."""
    C, H, Wd = shape
    T = Wfull.shape[1]
    A = np.abs(Wfull).reshape(C, H, Wd, T)
    mass = A.sum(0)                                                   # (H, W, T)
    tot = mass.reshape(-1, T).sum(0)

    def stats(ms):
        tot_ = ms.reshape(-1, ms.shape[-1]).sum(0)
        best1 = ms.reshape(-1, ms.shape[-1]).max(0)
        win = ms[:-1, :-1] + ms[1:, :-1] + ms[:-1, 1:] + ms[1:, 1:]
        best4 = win.reshape(-1, ms.shape[-1]).max(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return best1 / tot_, best4 / tot_
    f1, f4 = stats(mass)
    k = mass.reshape(-1, T).argmax(0)
    peak = np.stack([k // Wd, k % Wd], 1)
    rng = np.random.default_rng(seed)
    n1, n4 = np.zeros((n_null, T)), np.zeros((n_null, T))
    for t in range(T):
        c, h, w = np.nonzero(A[..., t])
        v = A[c, h, w, t]
        if v.size == 0:
            n1[:, t] = n4[:, t] = np.nan
            continue
        for i in range(n_null):
            pos = rng.integers(0, H * Wd, v.size)
            ms = np.bincount(pos, weights=v, minlength=H * Wd).reshape(H, Wd, 1)
            a, b = stats(ms)
            n1[i, t], n4[i, t] = a[0], b[0]
    # centre of mass (row, col)
    hh, ww = np.meshgrid(np.arange(H), np.arange(Wd), indexing="ij")
    with np.errstate(invalid="ignore", divide="ignore"):
        com = np.stack([(mass * hh[..., None]).reshape(-1, T).sum(0) / tot,
                        (mass * ww[..., None]).reshape(-1, T).sum(0) / tot], 1)
    return dict(frac1=f1, frac4=f4, null1=np.nanmean(n1, 0), null4=np.nanmean(n4, 0), peak=peak, com=com,
                n_positions=(mass.reshape(-1, T) > 0).sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="unit-100k")
    ap.add_argument("--members", default="1,2,3,4,5")
    ap.add_argument("--skip-cv", action="store_true")
    a = ap.parse_args()
    t_all = time.perf_counter()
    tier = load_tier(a.tier)
    shape = LAYER_MAPS[tier.meta["layer"]][0]
    p = int(np.prod(shape))
    kp = os.path.join(tier.path, "kept_columns.npy")
    keep = np.load(kp) if os.path.exists(kp) else np.arange(p)
    X, Y = np.asarray(tier.X), np.ascontiguousarray(tier.Y)
    Xte, Yte = np.asarray(tier.X_test), np.asarray(tier.Y_test, np.float64)
    rel = tier.reliable_mask()
    od = os.path.join(OUT_DIR, a.tier)
    os.makedirs(od, exist_ok=True)
    out, summ = {}, {}
    hw = runner.hardware_info()
    print("hardware", hw, flush=True)

    # 1. harness-protocol Lasso CV, coefficients kept
    if not a.skip_cv:
        row, res = runner.run("torch_ws", a.tier, task="CV", solver_args=dict(tol=1e-4),
                              out=os.path.join(od, "lasso_cv_rows.jsonl"), tag="factorized-readout-coefs")
        Wcv = full_w(res["coef"], keep, p)
        out["cv_coef"] = Wcv
        out["cv_r2"] = r2(Yte, Xte @ np.asarray(res["coef"].toarray() if sp.issparse(res["coef"]) else res["coef"]))
        summ["cv"] = dict(r2_rel=float(out["cv_r2"][rel].mean()), nnz_rel=float((Wcv[:, rel] != 0).sum(0).mean()),
                          nnz_all=float((Wcv != 0).sum(0).mean()), time_s=row.get("time_total_s"),
                          run_id=row["run_id"])
        print("CV", summ["cv"], flush=True)

    solver = runner.get_solver("torch_ws", tol=1e-4)
    # 2. path on all training rows
    alphas = api.alpha_grid(X, Y, runner.N_ALPHAS, runner.ALPHA_EPS)
    t0 = time.perf_counter()
    res = solver.fit_path(X, Y, alphas)
    t_path = time.perf_counter() - t0
    pr2, pnnz, pnnz_all = [], [], []
    for c in res.coefs:
        P = np.asarray((c.T @ Xte.T).T) if sp.issparse(c) else Xte @ c
        rr = r2(Yte, P)
        pr2.append(rr)
        nz = np.diff(sp.csc_matrix(c).indptr)
        pnnz.append(nz)
    out["path_r2"], out["path_nnz"], out["path_alphas"] = np.asarray(pr2), np.asarray(pnnz), alphas
    summ["path"] = dict(time_s=t_path, r2_rel=np.asarray(pr2)[:, rel].mean(1).tolist(),
                        nnz_rel=np.asarray(pnnz)[:, rel].mean(1).tolist())
    print(f"path {t_path:.1f}s", flush=True)

    # 3. 5-member protocol
    members = [int(s) for s in a.members.split(",")]
    M, T = len(members), Y.shape[1]
    Wm, Pm, val_paths = [], [], []
    t0 = time.perf_counter()
    for seed in members:
        tr, va = member_split(len(Y), seed)
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xc = np.ascontiguousarray(X[tr] - mx, np.float32)
        Yc = np.ascontiguousarray(Y[tr] - my, np.float32)
        al = api.alpha_grid(Xc, Yc, runner.N_ALPHAS, runner.ALPHA_EPS)
        rs = solver.fit_path(Xc, Yc, al)
        Xva = X[va] - mx
        vr = np.stack([r2(Y[va], np.asarray((c.T @ Xva.T).T) + my) for c in rs.coefs])     # (A, T)
        best = vr.argmax(0)
        W = np.stack([sp.csc_matrix(rs.coefs[best[t]])[:, t].toarray().ravel() for t in range(T)], 1)
        Wm.append(full_w(W, keep, p))
        Pm.append((Xte - mx) @ W + my)
        val_paths.append(vr)
        print(f"member {seed}: val R2 rel={vr[best, np.arange(T)][rel].mean():.3f} "
              f"test R2 rel={r2(Yte, Pm[-1])[rel].mean():.3f} nnz rel={(W[:, rel] != 0).sum(0).mean():.0f}", flush=True)
    t_mem = time.perf_counter() - t0
    Wm, Pm = np.stack(Wm), np.stack(Pm)                              # (M, p, T), (M, n_te, T)
    r2_ens = r2(Yte, Pm.mean(0))
    r2_mem = np.stack([r2(Yte, Pm[m]) for m in range(M)])
    nnz_m = (Wm != 0).sum(1)                                         # (M, T)
    union = (Wm != 0).any(0).sum(0)
    C, H, Wd = shape
    jac, cos, dpos = np.zeros(T), np.zeros(T), np.zeros(T)
    peaks = np.stack([concentration(Wm[m], shape, n_null=1)["peak"] for m in range(M)])  # (M, T, 2)
    for t in range(T):
        js, cs, ds = [], [], []
        for i, j in itertools.combinations(range(M), 2):
            si, sj = Wm[i, :, t] != 0, Wm[j, :, t] != 0
            u = (si | sj).sum()
            js.append((si & sj).sum() / u if u else 1.0)
            ni, nj = np.linalg.norm(Wm[i, :, t]), np.linalg.norm(Wm[j, :, t])
            cs.append(Wm[i, :, t] @ Wm[j, :, t] / (ni * nj) if ni > 0 and nj > 0 else np.nan)
            ds.append(np.linalg.norm(peaks[i, t] - peaks[j, t]))
        jac[t], cos[t], dpos[t] = np.mean(js), np.nanmean(cs), np.mean(ds)
    out.update(member_W=Wm.astype(np.float32), member_pred=Pm, member_val_r2=np.stack(val_paths),
               member_r2_ens=r2_ens, member_jac=jac, member_cos=cos, member_dpos=dpos, member_peaks=peaks)
    m = dict(r2_test_mean=float(r2_ens.mean()), r2_test_mean_reliable=float(r2_ens[rel].mean()),
             r2_test_median_reliable=float(np.median(r2_ens[rel])),
             r2_test_mean_reliable_single_member=float(r2_mem[:, rel].mean()),
             r2_test_mean_reliable_single_member_sd=float(r2_mem[:, rel].mean(1).std()),
             nnz_mean=float(nnz_m.mean()), nnz_mean_reliable=float(nnz_m[:, rel].mean()),
             nnz_union_mean_reliable=float(union[rel].mean()), n_params_per_channel=float(len(keep)),
             stab_jaccard_mean_reliable=float(jac[rel].mean()), stab_jaccard_median_reliable=float(np.median(jac[rel])),
             stab_cosine_mean_reliable=float(np.nanmean(cos[rel])),
             stab_position_dist_px_mean_reliable=float(dpos[rel].mean()),
             stab_position_dist_px_median_reliable=float(np.median(dpos[rel])), n_reliable=int(rel.sum()),
             time_members_s=t_mem)
    summ["members"] = m
    print("members", json.dumps(m), flush=True)

    # 4. spatial concentration
    for key, W in (("cv", out.get("cv_coef")), ("member0", Wm[0]), ("path_mid", None)):
        if W is None:
            continue
        c = concentration(W, shape)
        for k_, v in c.items():
            out[f"conc_{key}_{k_}"] = v
        summ[f"conc_{key}"] = {k_: float(np.nanmean(v[rel])) for k_, v in c.items() if k_ in
                               ("frac1", "frac4", "null1", "null4", "n_positions")}
        print("concentration", key, summ[f"conc_{key}"], flush=True)
    summ["hardware"] = hw
    summ["time_script_s"] = time.perf_counter() - t_all
    np.savez_compressed(os.path.join(od, "lasso.npz"), **out)
    with open(os.path.join(od, "lasso_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    row = dict(run_id=time.strftime("%Y%m%d-%H%M%S-") + "lassom", timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
               task="CV", solver="fact_lasso_members", tier=a.tier,
               shape=dict(n_train=int(X.shape[0]), n_test=int(Xte.shape[0]), p=int(X.shape[1]), T=int(T)),
               max_targets=None, tag="factorized-readout",
               config=dict(device="cuda", base_solver="torch_ws", tol=1e-4, members=members,
                           selection="per-channel alpha by member val R2; 5-member ensemble (80/20 member splits)",
                           n_alphas=runner.N_ALPHAS, alpha_eps=runner.ALPHA_EPS),
               hardware=hw, **runner.git_info(), status="ok", time_total_s=t_mem, time_prepare_s=0.0,
               time_solve_s=t_mem, metrics=m, raw=os.path.join(od, "lasso.npz"))
    runner.append_row(row)


if __name__ == "__main__":
    main()
