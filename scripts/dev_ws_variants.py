"""Sparse variants of torch_ws on a tier: CV time, reliable-channel test R2 at the selected alpha (min / 1se),
nnz, fold-support Jaccard, and the R2-vs-nnz frontier at k = 10, 30, 100, 300, 1000 from the full-data path
(R2 interpolated linearly in log(nnz) between the two path points that bracket k, per reliable channel).

Usage: python scripts/dev_ws_variants.py tier [max_targets] [variant ...]
Variants: lasso, enet0.5, enet0.9, relax0.01, relax0.1, enet0.5+relax0.01  (select min and 1se both reported)
"""
import json
import sys
import time

import numpy as np
import torch

from lassobench import runner
from lassobench.api import STORE, alpha_grid
from lassobench.data import load_tier
from lassobench.solvers.torch_ws import TorchWS

tier = sys.argv[1]
mt = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None
variants = sys.argv[3:] or ["lasso", "enet0.5", "relax0.01"]
d = load_tier(tier)
X, Y = np.asarray(d.X), np.asarray(d.Y)
Xt, Yt = np.asarray(d.X_test), np.asarray(d.Y_test, np.float64)
rel = d.reliable_mask()
if mt:
    Y, Yt, rel = Y[:, :mt], Yt[:, :mt], rel[:mt]
T = Y.shape[1]
alphas = alpha_grid(X, Y, 50, 1e-2)
folds = runner.make_folds(X.shape[0])
KS = [10, 30, 100, 300, 1000]


def kw_of(v):
    kw = {}
    for part in v.split("+"):
        if part.startswith("enet"):
            kw["l1_ratio"] = float(part[4:])
        elif part == "relaxcv":
            kw["relax"] = "cv"
        elif part.startswith("relax"):
            kw["relax_gamma"] = float(part[5:])
    return kw


def r2(W):
    P = Xt @ W
    return 1 - ((Yt - P) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b)) if (a or b) else 1.0


TorchWS(tol=1e-4).fit_path(X[:, :2000], Y[:, :4], alphas[:5, :4])   # warm-up
for v in variants:
    kw = kw_of(v)
    for sel in ["min", "1se"]:
        s = TorchWS(tol=1e-4, select=sel, **kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = s.fit_cv(X, Y, alphas, folds)
        t_cv = time.perf_counter() - t0
        W = r["coef"].toarray()
        rr = r2(W)
        nnz = (W != 0).sum(0)
        fc = r["fold_coefs"]
        jac = np.array([np.mean([jacc(fc[i][:, t].indices, fc[j][:, t].indices)
                                 for i in range(len(fc)) for j in range(i + 1, len(fc))]) for t in range(T)])
        row = dict(tier=tier, T=T, variant=v, select=sel, kw=kw, gpu=torch.cuda.get_device_name(0), time_cv_s=t_cv,
                   r2_rel=float(rr[rel].mean()), r2_all=float(rr.mean()), nnz_median_rel=float(np.median(nnz[rel])),
                   nnz_mean_rel=float(nnz[rel].mean()), fold_jaccard_rel_mean=float(jac[rel].mean()),
                   fold_jaccard_rel_median=float(np.median(jac[rel])),
                   alpha_stop_mean=r["info"]["fold_infos"][0].get("alpha_stop_mean"),
                   timings_folds={k: round(x, 1) for k, x in r["info"]["fold_infos"][0]["timings"].items() if x > 0.5})
        if "relax" in r["info"]:
            WL = r["coef_lasso"].toarray()
            rl_ = r2(WL)
            ri = r["info"]["relax"]
            row.update(lasso_same_run_r2_rel=float(rl_[rel].mean()), lasso_same_run_nnz_median_rel=float(np.median((WL != 0).sum(0)[rel])),
                       time_relax_cv_s=ri["time_relax_cv_s"], time_relax_final_s=ri.get("time_relax_final_s"),
                       frac_no_refit=ri["frac_no_refit"], penalty_hist=np.bincount(ri["penalty_idx"], minlength=len(ri["grid"]) + 1).tolist())
            pen_per_alpha = np.array(ri["penalty_idx_per_alpha"])          # (A, T)
        print(json.dumps(row), flush=True)
        open(f"{STORE}/results/ws_dev/variants.jsonl", "a").write(json.dumps(row) + "\n")
        torch.cuda.empty_cache()
    # frontier from the full-data path
    s = TorchWS(tol=1e-4, **kw)
    t0 = time.perf_counter()
    res = s.fit_path(X, Y, alphas)
    t_p = time.perf_counter() - t0
    if kw.get("relax") == "cv":
        from lassobench.solvers.torch_ws import RELAX_GRID, _support_ridge, _triplets_to_supports
        dev = torch.device("cuda")
        Xg = torch.from_numpy(X).to(dev)
        Yg = torch.from_numpy(Y).to(dev)
        allr = torch.arange(X.shape[0], device=dev)
        newc = []
        for a, c in enumerate(res.coefs):
            cc = c.tocoo()
            ids, lv, valid = _triplets_to_supports(cc.row, cc.col, cc.data, list(range(T)), dev)
            bet, _ = _support_ridge(Xg, Yg, allr, None, ids, valid, RELAX_GRID)
            li = torch.as_tensor(pen_per_alpha[a], device=dev).long()
            ch = bet[li.clamp(max=len(RELAX_GRID) - 1), torch.arange(T, device=dev)]
            ch = torch.where((li == len(RELAX_GRID))[:, None], lv, ch) * valid
            tt = torch.arange(T, device=dev)[:, None].expand_as(ids)
            import scipy.sparse as sp_
            newc.append(sp_.csc_matrix((ch[valid].cpu().numpy(), (ids[valid].cpu().numpy(), tt[valid].cpu().numpy())), shape=c.shape))
        res.coefs = newc
        del Xg, Yg
    R2 = np.array([r2(c.toarray()) for c in res.coefs])          # (A, T)
    NZ = np.array([np.diff(c.indptr) for c in res.coefs])         # (A, T)
    fr = {}
    for k in KS:
        vals = []
        for t in np.nonzero(rel)[0]:
            nz, rt = NZ[:, t].astype(float), R2[:, t]
            o = np.argsort(nz, kind="stable")
            nz, rt = nz[o], rt[o]
            if k < nz[0] or k > nz[-1]:
                continue
            lo = np.searchsorted(nz, k, side="right") - 1
            hi = min(lo + 1, len(nz) - 1)
            if nz[hi] == nz[lo]:
                vals.append(rt[lo])
            else:
                f = (np.log(k) - np.log(max(nz[lo], 1))) / (np.log(nz[hi]) - np.log(max(nz[lo], 1)))
                vals.append(rt[lo] + f * (rt[hi] - rt[lo]))
        fr[f"r2_rel_at_{k}"] = float(np.mean(vals)) if vals else None
        fr[f"n_ch_at_{k}"] = len(vals)
    row = dict(tier=tier, T=T, variant=v, kind="frontier", kw=kw, gpu=torch.cuda.get_device_name(0), time_path_s=t_p, **fr)
    print(json.dumps(row), flush=True)
    open(f"{STORE}/results/ws_dev/variants.jsonl", "a").write(json.dumps(row) + "\n")
    torch.cuda.empty_cache()
