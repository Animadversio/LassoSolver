"""Round 2: is the low-rank space x feature readout (factgd) really better than Ridge, and is it stable?

  python -m lassobench.factorized.round2 --tier unit-100k --parts splits,stab,ablate
  python -m lassobench.factorized.round2 --tier unit-800k --parts timing

Parts (each saves STORE/results/factorized/<tier>/round2/<part>[_k].npz; `summarize_round2.py` builds tables):

splits  5 train/test splits of the tier's 1028 images (split 0 = the tier's own split; split k >= 1 =
        RandomState(1000 + k) permutation, same test size, X columns and Y re-z-scored on that split's training
        rows). In each split, with every choice made inside the training part:
          RidgeCV (harness cpu_ridgecv: per-target LOO-GCV), Ridge bagged over 5 member 80% subsets,
          Lasso CV (torch_ws fit_cv, harness folds/grid, min-CV alpha, one refit on all training rows),
          Lasso 5-member ensemble (per-channel alpha by member validation R2),
          factgd rank 1, 2: penalty grid x 5 members (member m: RandomState(m) 80/20 split, init seed m) ->
          per-channel gamma* by mean member validation R2 (pruned model); then the 5 members retrained at the
          per-channel gamma* vector (ensemble), and one refit on 100% of the training rows at gamma* with the LR
          schedule and epoch count of member 1's validated run (single model).
stab    split 0 only. Two separate sources of variability, each at the selected penalty and fixed schedule:
          (a) seed: same rows (100% of training), init / batch-order seeds 1..8;
          (b) data: init seed 1, five RandomState(m) 80% subsets and the two disjoint halves of the training rows.
        Same for Ridge (per-target alpha from RidgeCV on all rows, alpha scaled by n_sub / n) and Lasso (per-target
        CV alpha, 1/(2n) objective so alpha needs no rescaling); both are deterministic, so (a) does not apply.
        Saved: test predictions and factors / weights for the stability tables.
ablate  split 0 only: rank 3; L1 on features only / on masks only / none; non-negative mask; smooth mask
        (scale-free roughness penalty); convex stand-ins: ridge then truncated SVD of each channel's C x HW
        weight matrix (rank 1-3), and nuclear-norm-penalized regression (FISTA + singular-value thresholding,
        per-channel lambda by member validation R2, refit on 100%).
timing  one factgd rank-1 / rank-2 member on a tier (for unit-800k: time and peak GPU memory).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

from lassobench.factorized.data import OUT_DIR, load_maps, member_split
from lassobench.factorized.models import FactorizedReadout
from lassobench.factorized.run import GAMMAS, prune_rel
from lassobench.factorized.train import predict, r2_cols, train_member

N_SPLITS, SPLIT_SEED0 = 5, 1000
MEMBERS = [1, 2, 3, 4, 5]
STAB_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
SMOOTH = [1.0, 3.0, 10.0, 30.0]
SCALE_GAMMAS = [150.0, 220.0, 330.0, 500.0, 1000.0]
NUC_GRID = np.logspace(0, -2.5, 14)
NUC_ITERS = 150
QUICK = [False]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def r2(Yt, P):
    Yt = np.asarray(Yt, np.float64)
    return 1 - ((Yt - P) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)


# ----------------------------------------------------------------------------------------------- data / splits

class Split:
    def __init__(self, k, Xtr, Ytr, Xte, Yte, rel, shape, tr_idx, te_idx):
        self.k, self.Xtr, self.Ytr, self.Xte, self.Yte = k, Xtr, Ytr, Xte, Yte
        self.rel, self.shape, self.tr_idx, self.te_idx = rel, tuple(shape), tr_idx, te_idx

    @property
    def n(self):
        return len(self.Ytr)

    def flat(self, X):
        return X.reshape(len(X), -1)


def load_splits(tier, ks):
    d = load_maps(tier, raw=False)
    tr0, te0 = np.asarray(d.meta["train_idx"]), np.asarray(d.meta["test_idx"])
    N = len(tr0) + len(te0)
    out = {}
    for k in ks:
        if k == 0:
            out[0] = Split(0, d.Xz, d.Y, d.Xz_test, d.Y_test, d.reliable, d.shape, tr0, te0)
            continue
        Xall = np.empty((N, d.p), np.float32)
        Xall[tr0], Xall[te0] = d.Xz.reshape(len(tr0), -1), d.Xz_test.reshape(len(te0), -1)
        Yall = np.empty((N, d.Y.shape[1]), np.float32)
        Yall[tr0], Yall[te0] = d.Y, d.Y_test
        perm = np.random.RandomState(SPLIT_SEED0 + k).permutation(N)
        te, tr = np.sort(perm[:len(te0)]), np.sort(perm[len(te0):])
        mu, sd = Xall[tr].mean(0), Xall[tr].std(0)
        ok = sd > 1e-6
        sd = np.where(ok, sd, 1.0)
        Xtr = ((Xall[tr] - mu) / sd) * ok
        Xte = ((Xall[te] - mu) / sd) * ok
        ym, ys = Yall[tr].mean(0), Yall[tr].std(0)
        out[k] = Split(k, Xtr.astype(np.float32).reshape(-1, *d.shape), (Yall[tr] - ym) / ys,
                       Xte.astype(np.float32).reshape(-1, *d.shape), (Yall[te] - ym) / ys, d.reliable, d.shape,
                       tr, te)
        del Xall
    return d, out


# ----------------------------------------------------------------------------------------------- Ridge / Lasso

def ridgecv(X, Y):
    """Harness RidgeCV (sklearn LOO-GCV per target, intercept). Returns W (p, T), b (T,), alpha (T,)."""
    from lassobench.solvers.cpu_ridgecv import RidgeCVBaseline
    r = RidgeCVBaseline().fit_cv(X, Y, None, None)
    W = r["coef"]
    alpha = r["alphas"][r["best_idx"], np.arange(Y.shape[1])]
    b = Y.mean(0) - X.mean(0) @ W
    return W, b, alpha


def ridge_fixed(X, Y, alpha):
    """Ridge at a given per-target alpha (sklearn scaling: |y - Xw|^2 + alpha |w|^2), intercept, GPU dual."""
    Xt = torch.as_tensor(X, device=DEV, dtype=torch.float64)
    Yt = torch.as_tensor(Y, device=DEV, dtype=torch.float64)
    mx, my = Xt.mean(0), Yt.mean(0)
    Xc, Yc = Xt - mx, Yt - my
    s, U = torch.linalg.eigh(Xc @ Xc.T)
    UY = U.T @ Yc                                                     # (n, T)
    a = torch.as_tensor(alpha, device=DEV, dtype=torch.float64)
    Wt = Xc.T @ (U @ (UY / (s[:, None] + a[None, :])))
    W = Wt.float().cpu().numpy()
    return W, (my - mx @ Wt).float().cpu().numpy()


def _solver():
    from lassobench import runner
    return runner.get_solver("torch_ws", tol=1e-4)


def lasso_cv(X, Y):
    """Harness-protocol Lasso CV (torch_ws.fit_cv, 5 folds seed 0, 50 alphas eps 1e-2, min CV MSE)."""
    from lassobench import api, runner
    al = api.alpha_grid(X, Y, runner.N_ALPHAS, runner.ALPHA_EPS)
    r = _solver().fit_cv(X, Y, al, runner.make_folds(X.shape[0]))
    W = r["coef"]
    W = W.toarray() if sp.issparse(W) else np.asarray(W)
    return W.astype(np.float32), al, np.asarray(r["best_idx"])


def lasso_fixed(X, Y, al, best):
    """Lasso at each target's alpha al[best[t], t], following the path from the top of the grid (centered)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc = np.ascontiguousarray(X - mx, np.float32)
    Yc = np.ascontiguousarray(Y - my, np.float32)
    res = _solver().fit_path(Xc, Yc, np.ascontiguousarray(al[: int(best.max()) + 1]))
    W = np.stack([sp.csc_matrix(res.coefs[best[t]])[:, t].toarray().ravel() for t in range(Y.shape[1])], 1)
    return W.astype(np.float32), (my - mx @ W).astype(np.float32)


def lasso_members(X, Y, Xte, members=None):
    """Per member: path on the member's 80% (centered), per-target alpha by member validation R2."""
    from lassobench import api, runner
    Ws, bs, P = [], [], []
    for m in members or MEMBERS:
        tr, va = member_split(len(Y), m)
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Xc = np.ascontiguousarray(X[tr] - mx, np.float32)
        Yc = np.ascontiguousarray(Y[tr] - my, np.float32)
        al = api.alpha_grid(Xc, Yc, runner.N_ALPHAS, runner.ALPHA_EPS)
        rs = _solver().fit_path(Xc, Yc, al)
        Xva = X[va] - mx
        vr = np.stack([r2(Y[va], np.asarray((sp.csc_matrix(c).T @ Xva.T).T) + my) for c in rs.coefs])
        best = vr.argmax(0)
        W = np.stack([sp.csc_matrix(rs.coefs[best[t]])[:, t].toarray().ravel() for t in range(Y.shape[1])], 1)
        b = my - mx @ W
        Ws.append(W.astype(np.float32)), bs.append(b), P.append(Xte @ W + b)
    return np.stack(Ws), np.stack(bs), np.stack(P)


# ----------------------------------------------------------------------------------------------- factgd

def _pen(variant, gam):
    """(gamma_f, gamma_s) for a penalty value under a variant's L1 placement."""
    l1 = variant.get("l1", "both")
    return {"both": (gam, gam), "f": (gam, 0.0), "s": (0.0, gam), "none": (0.0, 0.0)}[l1]


def _vec(x, T):
    return torch.as_tensor(np.broadcast_to(np.asarray(x, np.float32), (T,)).copy(), device=DEV)


def fit_fact(Xtr, Ytr, Xva, Yva, rank, gf, gs, init_seed, train_seed, variant, lr_schedule=None):
    """One factgd model (prox L1 on F with gf, on S with gs; floats or (T,) arrays). Returns (model, info)."""
    T = Ytr.shape[1]
    model = FactorizedReadout(tuple(Xtr.shape[1:]), T, rank=rank, seed=init_seed).to(DEV)
    if variant.get("nonneg"):
        with torch.no_grad():
            model.S.clamp_(min=0)
    sm = variant.get("smooth")
    info = train_member(model, Xtr, Ytr, Xva, Yva, loss="mse", gamma=_vec(gf, T), gamma_s=_vec(gs, T),
                        metric="r2", seed=train_seed, l1_mode="prox", nonneg_s=bool(variant.get("nonneg")),
                        smooth_s=None if sm is None else _vec(sm, T), lr_schedule=lr_schedule)
    return model, info


@torch.no_grad()
def pruned(model):
    return prune_rel(model.S.detach(), 2), prune_rel(model.F.detach(), 2)


@torch.no_grad()
def fact_eval(model, X):
    Spr, Fpr = pruned(model)
    return predict(model, X, S=Spr, Fv=Fpr)


def fact_record(model):
    Spr, Fpr = pruned(model)
    W = model.unit_weights(Spr, Fpr).reshape(-1, model.T)
    return dict(S=Spr.cpu().numpy(), F=Fpr.cpu().numpy(), bias=model.bias.detach().cpu().numpy(),
                eff_units=(W != 0).sum(0).cpu().numpy(), nnz_ch=(Fpr != 0).any(1).sum(1).cpu().numpy())


def penalty_grid(variant):
    if variant.get("l1") == "none":
        g = [0.0]
    else:
        g = list(variant.get("gammas", GAMMAS))
    sm = variant.get("smooth_grid")
    return [(x, None) for x in g] if sm is None else [(x, s) for s in sm for x in g]


def factgd_pipeline(sp_, rank, variant=None, name=""):
    """Selection grid x 5 members -> per-channel penalty -> retrained 5 members (ensemble) + 100% refit (single).

    Returns dict with test predictions (pruned), selected penalties, refit schedule, factors, timings.
    """
    variant = dict(variant or {})
    T = sp_.Ytr.shape[1]
    X = torch.as_tensor(sp_.Xtr, device=DEV)
    Y = torch.as_tensor(sp_.Ytr, device=DEV)
    Xte = torch.as_tensor(sp_.Xte, device=DEV)
    Yte = sp_.Yte
    grid = penalty_grid(variant)
    t0 = time.perf_counter()
    val = np.zeros((len(grid), len(MEMBERS), T))
    for gi, (gam, sm) in enumerate(grid):
        v = dict(variant, smooth=sm)
        gf, gs = _pen(variant, gam)
        for mi, m in enumerate(MEMBERS):
            tr, va = member_split(sp_.n, m)
            model, _ = fit_fact(X[tr], Y[tr], X[va], Y[va], rank, gf, gs, m, m, v)
            val[gi, mi] = r2_cols(fact_eval(model, X[va]), Y[va])
    t_grid = time.perf_counter() - t0
    mval = np.where(np.isfinite(val), val, -np.inf).mean(1)
    gsel = mval.argmax(0)
    gam_t = np.array([grid[g][0] for g in gsel], np.float32)
    sm_t = None if variant.get("smooth_grid") is None else np.array([grid[g][1] for g in gsel], np.float32)
    gf_t, gs_t = _pen(variant, gam_t) if variant.get("l1", "both") != "none" else (0.0, 0.0)
    if variant.get("l1") == "f":
        gs_t = 0.0
    if variant.get("l1") == "s":
        gf_t = 0.0
    vsel = dict(variant, smooth=sm_t)
    # members at the selected per-channel penalty
    t1 = time.perf_counter()
    mem = dict(pred=[], val_r2=[], best_epoch=[], lr_hist=[], S=[], F=[], eff_units=[], nnz_ch=[])
    for m in MEMBERS:
        tr, va = member_split(sp_.n, m)
        model, info = fit_fact(X[tr], Y[tr], X[va], Y[va], rank, gf_t, gs_t, m, m, vsel)
        rec = fact_record(model)
        mem["pred"].append(fact_eval(model, Xte).cpu().numpy())
        mem["val_r2"].append(r2_cols(fact_eval(model, X[va]), Y[va]))
        mem["best_epoch"].append(info["best_epoch"])
        mem["lr_hist"].append(info["lr_hist"])
        for k_ in ("S", "F", "eff_units", "nnz_ch"):
            mem[k_].append(rec[k_])
    t_mem = time.perf_counter() - t1
    sched = mem["lr_hist"][0][: mem["best_epoch"][0]]
    t2 = time.perf_counter()
    model, info = fit_fact(X, Y, None, None, rank, gf_t, gs_t, 1, 1, vsel, lr_schedule=sched)
    t_refit = time.perf_counter() - t2
    rec = fact_record(model)
    P_single = fact_eval(model, Xte).cpu().numpy()
    P_mem = np.stack(mem["pred"])
    out = dict(val_grid=val, grid=np.array([(g, -1 if s is None else s) for g, s in grid], float), gsel=gsel,
               gamma_f=np.broadcast_to(np.asarray(gf_t, np.float32), (T,)).copy(),
               gamma_s=np.broadcast_to(np.asarray(gs_t, np.float32), (T,)).copy(),
               smooth=np.full(T, np.nan, np.float32) if sm_t is None else sm_t, sched=np.asarray(sched),
               single_pred=P_single, single_S=rec["S"], single_F=rec["F"], single_bias=rec["bias"],
               single_eff_units=rec["eff_units"], single_nnz_ch=rec["nnz_ch"],
               mem_pred=P_mem, mem_val_r2=np.stack(mem["val_r2"]), mem_best_epoch=np.asarray(mem["best_epoch"]),
               mem_S=np.stack(mem["S"]), mem_F=np.stack(mem["F"]), mem_eff_units=np.stack(mem["eff_units"]),
               mem_nnz_ch=np.stack(mem["nnz_ch"]), time_grid_s=t_grid, time_members_s=t_mem,
               time_refit_s=t_refit)
    rel = sp_.rel
    _log(f"  {name} r{rank} split{sp_.k}: single R2 rel={r2(Yte, P_single)[rel].mean():.3f} "
         f"ens={r2(Yte, P_mem.mean(0))[rel].mean():.3f} member={np.mean([r2(Yte, p)[rel].mean() for p in P_mem]):.3f}"
         f" eff={rec['eff_units'][rel].mean():.0f} epochs={len(sched)} "
         f"(grid {t_grid:.0f}s, members {t_mem:.1f}s, refit {t_refit:.1f}s)")
    return out


def fact_fixed(sp_, rank, pipe, rows, init_seed, variant=None):
    """Retrain at a pipeline's selected penalty and schedule on a subset of training rows (no validation)."""
    variant = dict(variant or {})
    X = torch.as_tensor(sp_.Xtr[rows], device=DEV)
    Y = torch.as_tensor(sp_.Ytr[rows], device=DEV)
    sm = pipe["smooth"]
    v = dict(variant, smooth=None if np.all(np.isnan(sm)) else sm)
    model, info = fit_fact(X, Y, None, None, rank, pipe["gamma_f"], pipe["gamma_s"], init_seed, init_seed, v,
                           lr_schedule=list(pipe["sched"]))
    rec = fact_record(model)
    rec["pred"] = fact_eval(model, torch.as_tensor(sp_.Xte, device=DEV)).cpu().numpy()
    rec["time_s"] = info["time_s"]
    return rec


# ----------------------------------------------------------------------------------------------- convex stand-ins

def ridge_tsvd(W, b, shape, Xte, ranks=(1, 2, 3)):
    """Truncate each channel's Ridge weight matrix (C x HW) to rank r. Returns {r: (pred, W_r)}."""
    C, H, Wd = shape
    T = W.shape[1]
    Wm = torch.as_tensor(W.T.reshape(T, C, H * Wd), device=DEV)
    U, s, Vh = torch.linalg.svd(Wm, full_matrices=False)
    Xt = torch.as_tensor(Xte.reshape(len(Xte), -1), device=DEV)
    out = {}
    for r in ranks:
        Wr = torch.einsum("tcr,tr,trs->tcs", U[..., :r], s[:, :r], Vh[:, :r]).reshape(T, -1).T
        out[r] = ((Xt @ Wr).cpu().numpy() + b, Wr.cpu().numpy())
    return out


def _svt(V, thr):
    """Singular-value soft-thresholding of each (C x HW) slice of V (T, C, HW) by thr (T,), via the HW x HW Gram."""
    G = (V.transpose(1, 2) @ V).double()
    e, Q = torch.linalg.eigh(G)
    s = e.clamp(min=0).sqrt()
    f = ((s - thr.double()[:, None]).clamp(min=0) / s.clamp(min=1e-12)).float()
    Qf = Q.float()
    return V @ (Qf * f[:, None, :]) @ Qf.transpose(1, 2), (s > thr.double()[:, None]).sum(1)


def nuclear_path(X, Y, shape, lam_rel=None, iters=None, lam_max=None):
    """min_W 1/(2n)|Yc - Xc W|^2 + lam_t |W_t|_* per channel, W_t = C x HW; FISTA with restart, warm start
    down the grid lam_t = lam_rel * lam_max_t. X (n, p), Y (n, T) torch. Yields (k, W (p, T), b (T,), rank)."""
    C, H, Wd = shape
    lam_rel = NUC_GRID if lam_rel is None else lam_rel
    iters = NUC_ITERS if iters is None else iters
    n, p = X.shape
    T = Y.shape[1]
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    if lam_max is None:                                               # smallest lam with W = 0, per channel
        G0 = (Xc.T @ Yc / n).T.reshape(T, C, H * Wd)
        lam_max = torch.linalg.matrix_norm(G0, ord=2)                 # (T,)
    lam_max = torch.as_tensor(lam_max, device=X.device, dtype=torch.float32)
    v = torch.randn(p, device=X.device)
    for _ in range(30):
        v = Xc.T @ (Xc @ v)
        v = v / v.norm()
    L = float((Xc @ v).norm() ** 2 / n) * 1.05
    W = torch.zeros(T, C, H * Wd, device=X.device)
    for k, lr_ in enumerate(lam_rel):
        thr = lam_max * float(lr_) / L
        Z, tk = W.clone(), torch.ones(T, device=X.device)
        for it in range(iters):
            R = Xc @ Z.reshape(T, -1).T - Yc
            grad = (Xc.T @ R / n).T.reshape(T, C, H * Wd)
            Wn, rk = _svt(Z - grad / L, thr)
            restart = ((Z - Wn) * (Wn - W)).sum((1, 2)) > 0                 # O'Donoghue-Candes, per channel
            tk = torch.where(restart, torch.ones_like(tk), tk)
            tn = (1 + torch.sqrt(1 + 4 * tk * tk)) / 2
            Z = Wn + ((tk - 1) / tn)[:, None, None] * (Wn - W)
            W, tk = Wn, tn
        Wf = W.reshape(T, -1).T
        yield k, Wf, my - mx @ Wf, rk


def nuclear_pipeline(sp_, name="nuclear"):
    T = sp_.Ytr.shape[1]
    X = torch.as_tensor(sp_.flat(sp_.Xtr), device=DEV)
    Y = torch.as_tensor(sp_.Ytr, device=DEV)
    Xte = torch.as_tensor(sp_.flat(sp_.Xte), device=DEV)
    t0 = time.perf_counter()
    val = np.zeros((len(NUC_GRID), len(MEMBERS), T))
    preds = np.zeros((len(NUC_GRID), len(MEMBERS), len(Xte), T), np.float32)
    ranks = np.zeros((len(NUC_GRID), len(MEMBERS), T))
    for mi, m in enumerate(MEMBERS):
        tr, va = member_split(sp_.n, m)
        for k, W, b, rk in nuclear_path(X[tr], Y[tr], sp_.shape):
            val[k, mi] = r2_cols(X[va] @ W + b, Y[va])
            preds[k, mi] = (Xte @ W + b).cpu().numpy()
            ranks[k, mi] = rk.cpu().numpy()
    gsel = val.mean(1).argmax(0)
    tt = np.arange(T)
    P_mem = preds[gsel, :, :, tt].transpose(1, 2, 0)                  # (M, n_te, T)
    # refit on 100%: run the path and take each channel's selected point
    Wsel, bsel, rsel = nuclear_at(X, Y, sp_.shape, gsel)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    lam_max = torch.linalg.matrix_norm((Xc.T @ Yc / sp_.n).T.reshape(T, sp_.shape[0], -1), ord=2)
    del Xc, Yc
    P_single = (Xte @ Wsel + bsel).cpu().numpy()
    rel = sp_.rel
    _log(f"  {name} split{sp_.k}: single R2 rel={r2(sp_.Yte, P_single)[rel].mean():.3f} "
         f"ens={r2(sp_.Yte, P_mem.mean(0))[rel].mean():.3f} rank rel={rsel[rel].mean():.2f} "
         f"({time.perf_counter() - t0:.0f}s)")
    return dict(val_grid=val, gsel=gsel, single_pred=P_single, mem_pred=P_mem, single_rank=rsel,
                mem_rank=ranks[gsel, :, tt].T, single_W=Wsel.cpu().numpy(), lam_max=lam_max.cpu().numpy(),
                time_s=time.perf_counter() - t0)


def nuclear_at(X, Y, shape, gsel, lam_max=None):
    """Nuclear-norm fit stopping each channel at grid point gsel[t] (lam = NUC_GRID[gsel] * lam_max)."""
    T = Y.shape[1]
    Wsel = torch.zeros(X.shape[1], T, device=X.device)
    bsel = torch.zeros(T, device=X.device)
    rsel = np.zeros(T)
    for k, W, b, rk in nuclear_path(X, Y, shape, lam_rel=NUC_GRID[: int(gsel.max()) + 1], lam_max=lam_max):
        cols = np.flatnonzero(gsel == k)
        Wsel[:, cols], bsel[cols] = W[:, cols], b[cols]
        rsel[cols] = rk.cpu().numpy()[cols]
    return Wsel, bsel, rsel


def part_nuclear(tier, splits):
    """Nuclear-norm stand-in on every split (as `splits`) and, on split 0, its seed-free data stability:
    fixed per-channel lambda (absolute, from the all-rows fit) on the five 80% subsets and the two halves."""
    from lassobench.runner import half_split
    for k, sp_ in splits.items():
        t0 = time.perf_counter()
        nuc = nuclear_pipeline(sp_)
        res = {f"nuclear_{k_}": v for k_, v in nuc.items() if k_ != "single_W"}
        if k == 0:
            res["nuclear_single_W"] = nuc["single_W"]
            n = sp_.n
            subsets = [("sub%d" % m, member_split(n, m)[0]) for m in MEMBERS]
            subsets += [("half%d" % i, h) for i, h in enumerate(half_split(n, seed=1))]
            X = torch.as_tensor(sp_.flat(sp_.Xtr), device=DEV)
            Y = torch.as_tensor(sp_.Ytr, device=DEV)
            Xte = torch.as_tensor(sp_.flat(sp_.Xte), device=DEV)
            Ws, Ps, Rk = [], [], []
            for nm, rows in subsets:
                ri = torch.as_tensor(rows, device=DEV)
                W, b, rk = nuclear_at(X[ri], Y[ri], sp_.shape, nuc["gsel"], lam_max=nuc["lam_max"])
                Ws.append(W.cpu().numpy()), Ps.append((Xte @ W + b).cpu().numpy()), Rk.append(rk)
                _log(f"  nuclear stab {nm}: R2 rel={r2(sp_.Yte, Ps[-1])[sp_.rel].mean():.3f} "
                     f"rank={rk[sp_.rel].mean():.2f}")
            res.update(nuclear_data_W=np.stack(Ws), nuclear_data_pred=np.stack(Ps), nuclear_data_rank=np.stack(Rk),
                       subset_names=np.array([s for s, _ in subsets]))
        _save(tier, f"nuclear_split{k}", meta=_meta(tier, f"nuclear_split{k}", dict(time_s=time.perf_counter() - t0)),
              Yte=sp_.Yte, reliable=sp_.rel, **res)


# ----------------------------------------------------------------------------------------------- parts

def _save(tier, part, **arrs):
    od = os.path.join(OUT_DIR, tier, "round2_quick" if QUICK[0] else "round2")
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"{part}.npz")
    np.savez_compressed(fp, **arrs)
    _log(f"saved {fp}")


def _meta(tier, part, extra=None):
    from lassobench.runner import git_info, hardware_info
    hw = hardware_info()
    if DEV.type == "cuda":
        hw["gpu_model_torch"] = torch.cuda.get_device_name()
        hw["gpu_model"] = hw.get("gpu_model") or hw["gpu_model_torch"]
    return json.dumps(dict(tier=tier, part=part, hardware=hw, **git_info(), argv=sys.argv,
                           timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})), default=str)


def part_split(tier, sp_):
    X, Xte = sp_.flat(sp_.Xtr), sp_.flat(sp_.Xte)
    rel = sp_.rel
    res, tm = {}, {}
    t0 = time.perf_counter()
    W, b, alpha = ridgecv(X, sp_.Ytr)
    tm["ridgecv"] = time.perf_counter() - t0
    res["ridgecv_pred"], res["ridgecv_alpha"] = Xte @ W + b, alpha
    t0 = time.perf_counter()
    Pb = []
    for m in MEMBERS:
        tr, _ = member_split(sp_.n, m)
        Wm, bm, _ = ridgecv(X[tr], sp_.Ytr[tr])
        Pb.append(Xte @ Wm + bm)
    tm["ridge_bag"] = time.perf_counter() - t0
    res["ridgebag_mem_pred"] = np.stack(Pb)
    t0 = time.perf_counter()
    Wl, al, best = lasso_cv(X, sp_.Ytr)
    tm["lasso_cv"] = time.perf_counter() - t0
    res["lassocv_pred"], res["lassocv_nnz"] = Xte @ Wl, (Wl != 0).sum(0)
    res["lassocv_alpha"] = al[best, np.arange(len(best))]
    t0 = time.perf_counter()
    Wm, bm, Pm = lasso_members(X, sp_.Ytr, Xte)
    tm["lasso_mem"] = time.perf_counter() - t0
    res["lassomem_pred"], res["lassomem_nnz"] = Pm, (Wm != 0).sum(1)
    _log(f"split{sp_.k}: ridgecv {r2(sp_.Yte, res['ridgecv_pred'])[rel].mean():.3f} "
         f"ridge-bag {r2(sp_.Yte, res['ridgebag_mem_pred'].mean(0))[rel].mean():.3f} "
         f"lasso-cv {r2(sp_.Yte, res['lassocv_pred'])[rel].mean():.3f} "
         f"lasso-mem {r2(sp_.Yte, Pm.mean(0))[rel].mean():.3f} times {json.dumps({k: round(v, 1) for k, v in tm.items()})}")
    for rank in (1, 2):
        pipe = factgd_pipeline(sp_, rank, name="factgd")
        for k_, v in pipe.items():
            if k_ not in ("mem_S", "mem_F", "val_grid") or sp_.k == 0:
                res[f"r{rank}_{k_}"] = v
    _save(tier, f"split{sp_.k}", meta=_meta(tier, f"split{sp_.k}", dict(times=tm)), Yte=sp_.Yte,
          te_idx=sp_.te_idx, tr_idx=sp_.tr_idx, reliable=rel, **res)


def part_stab(tier, sp_):
    X, Xte = sp_.flat(sp_.Xtr), sp_.flat(sp_.Xte)
    n = sp_.n
    from lassobench.runner import half_split
    subsets = [("sub%d" % m, member_split(n, m)[0]) for m in MEMBERS]
    subsets += [("half%d" % i, h) for i, h in enumerate(half_split(n, seed=1))]
    res = dict(subset_names=np.array([s for s, _ in subsets]))
    # Ridge / Lasso at the all-rows CV penalties, on each subset
    W, b, alpha = ridgecv(X, sp_.Ytr)
    Wl, al, best = lasso_cv(X, sp_.Ytr)
    res.update(ridge_full_W=W, ridge_full_pred=Xte @ W + b, ridge_alpha=alpha, lasso_full_W=Wl,
               lasso_full_pred=Xte @ Wl)
    rW, rP, lW, lP = [], [], [], []
    for nm, rows in subsets:
        Wr, br = ridge_fixed(X[rows], sp_.Ytr[rows], alpha * len(rows) / n)
        rW.append(Wr), rP.append(Xte @ Wr + br)
        Wq, bq = lasso_fixed(X[rows], sp_.Ytr[rows], al, best)
        lW.append(Wq), lP.append(Xte @ Wq + bq)
        _log(f"  stab {nm}: ridge {r2(sp_.Yte, rP[-1])[sp_.rel].mean():.3f} lasso "
             f"{r2(sp_.Yte, lP[-1])[sp_.rel].mean():.3f} nnz {(Wq[:, sp_.rel] != 0).sum(0).mean():.0f}")
    res.update(ridge_data_W=np.stack(rW).astype(np.float32), ridge_data_pred=np.stack(rP),
               lasso_data_W=np.stack(lW), lasso_data_pred=np.stack(lP))
    # factgd rank 1..3
    allrows = np.arange(n)
    for rank in (1, 2, 3):
        pipe = factgd_pipeline(sp_, rank, name="factgd")
        for k_ in ("gamma_f", "gamma_s", "sched", "single_pred"):
            res[f"r{rank}_{k_}"] = pipe[k_]
        for src, runs in (("seed", [(s, allrows) for s in STAB_SEEDS]), ("data", [(1, rows) for _, rows in subsets])):
            recs = [fact_fixed(sp_, rank, pipe, rows, s) for s, rows in runs]
            for k_ in ("S", "F", "pred", "eff_units"):
                res[f"r{rank}_{src}_{k_}"] = np.stack([r[k_] for r in recs])
            _log(f"  stab factgd r{rank} {src}: R2 rel per run "
                 f"{np.round([r2(sp_.Yte, r['pred'])[sp_.rel].mean() for r in recs], 3).tolist()}")
    _save(tier, "stab", meta=_meta(tier, "stab", dict(stab_seeds=STAB_SEEDS)), Yte=sp_.Yte, reliable=sp_.rel,
          **res)


ABLATIONS = {
    "r3": (3, {}),
    "r1_l1f": (1, {"l1": "f"}), "r1_l1s": (1, {"l1": "s"}), "r1_nol1": (1, {"l1": "none"}),
    "r2_l1f": (2, {"l1": "f"}), "r2_l1s": (2, {"l1": "s"}), "r2_nol1": (2, {"l1": "none"}),
    "r1_nonneg": (1, {"nonneg": True}), "r2_nonneg": (2, {"nonneg": True}),
    "r1_smooth": (1, {"smooth_grid": SMOOTH}), "r1_nonneg_smooth": (1, {"nonneg": True, "smooth_grid": SMOOTH}),
}


def part_ablate(tier, sp_, only=None):
    res = {}
    for name, (rank, var) in ABLATIONS.items():
        if only and name not in only:
            continue
        pipe = factgd_pipeline(sp_, rank, var, name=name)
        for k_ in ("single_pred", "mem_pred", "single_eff_units", "single_nnz_ch", "mem_eff_units", "gamma_f",
                   "gamma_s", "smooth", "single_S", "single_F", "time_grid_s"):
            res[f"{name}_{k_}"] = pipe[k_]
    X, Xte = sp_.flat(sp_.Xtr), sp_.flat(sp_.Xte)
    W, b, _ = ridgecv(X, sp_.Ytr)
    res["ridge_pred"] = Xte @ W + b
    for r, (P, Wr) in ridge_tsvd(W, b, sp_.shape, sp_.Xte).items():
        res[f"ridge_tsvd{r}_pred"] = P
        _log(f"  ridge-tsvd r{r}: R2 rel={r2(sp_.Yte, P)[sp_.rel].mean():.3f}")
    nuc = nuclear_pipeline(sp_)
    for k_, v in nuc.items():
        if k_ != "single_W":
            res[f"nuclear_{k_}"] = v
    _save(tier, "ablate", meta=_meta(tier, "ablate"), Yte=sp_.Yte, reliable=sp_.rel, **res)


def part_timing(tier, sp_):
    X = torch.as_tensor(sp_.Xtr, device=DEV)
    Y = torch.as_tensor(sp_.Ytr, device=DEV)
    out = {}
    tr, va = member_split(sp_.n, 1)
    fit_fact(X[tr], Y[tr], X[va], Y[va], 1, 4.5, 4.5, 1, 1, {})           # CUDA warm-up (not timed)
    for rank in (1, 2):
        for gam in (4.5,):
            tr, va = member_split(sp_.n, 1)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            model, info = fit_fact(X[tr], Y[tr], X[va], Y[va], rank, gam, gam, 1, 1, {})
            dt = time.perf_counter() - t0
            P = fact_eval(model, torch.as_tensor(sp_.Xte, device=DEV)).cpu().numpy()
            rec = fact_record(model)
            out[f"r{rank}"] = dict(time_member_s=dt, epochs=info["epochs"], s_per_epoch=dt / info["epochs"],
                                   peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                                   test_r2_rel=float(r2(sp_.Yte, P)[sp_.rel].mean()),
                                   eff_units_rel=float(rec["eff_units"][sp_.rel].mean()),
                                   protocol_estimate_s=dt * (len(GAMMAS) * 5 + 5 + 1))
            _log(f"timing {tier} r{rank}: {json.dumps(out[f'r{rank}'])}")
    # full selection protocol (grid x 5 members, retrained members, refit) on an extended gamma grid
    arrs = {}
    for rank in (1, 2):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pipe = factgd_pipeline(sp_, rank, {"gammas": GAMMAS + SCALE_GAMMAS}, name="scale")
        dt = time.perf_counter() - t0
        out[f"r{rank}_protocol"] = dict(time_s=dt, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                                        n_fits=len(GAMMAS + SCALE_GAMMAS) * len(MEMBERS) + len(MEMBERS) + 1,
                                        single_r2_rel=float(r2(sp_.Yte, pipe["single_pred"])[sp_.rel].mean()),
                                        ens_r2_rel=float(r2(sp_.Yte, pipe["mem_pred"].mean(0))[sp_.rel].mean()),
                                        single_eff_units_rel=float(pipe["single_eff_units"][sp_.rel].mean()),
                                        frac_at_top_gamma=float(np.mean(pipe["gsel"] == len(GAMMAS + SCALE_GAMMAS) - 1)))
        arrs[f"r{rank}_gsel"] = pipe["gsel"]
        _log(f"protocol {tier} r{rank}: {json.dumps(out[f'r{rank}_protocol'])}")
    _save(tier, "timing", meta=_meta(tier, "timing", dict(result=out, data_bytes_gpu=X.numel() * 4)), **arrs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--parts", default="splits,stab,ablate")
    ap.add_argument("--splits", default=",".join(str(k) for k in range(N_SPLITS)))
    ap.add_argument("--only", default=None, help="comma list of ablation names")
    ap.add_argument("--quick", action="store_true", help="tiny grids, 2 members (smoke test; not for results)")
    a = ap.parse_args(argv)
    if a.quick:
        global GAMMAS, MEMBERS, STAB_SEEDS, SMOOTH, NUC_GRID, NUC_ITERS
        GAMMAS, MEMBERS, STAB_SEEDS, SMOOTH = [0.0, 4.5], [1, 2], [1, 2], [3.0]
        NUC_GRID, NUC_ITERS = np.logspace(0, -2, 3), 20
        QUICK[0] = True
        _log("QUICK MODE")
    parts = a.parts.split(",")
    ks = [int(k) for k in a.splits.split(",")] if ("splits" in parts or "nuclear" in parts) else []
    if any(p in parts for p in ("stab", "ablate", "timing")) or "nuclear" in parts:
        ks = sorted(set(ks) | {0})
    t0 = time.perf_counter()
    d, S = load_splits(a.tier, ks)
    _log(f"loaded {a.tier} splits {ks} ({time.perf_counter() - t0:.0f}s) on {DEV} "
         f"{torch.cuda.get_device_name() if DEV.type == 'cuda' else ''}")
    if "timing" in parts:
        part_timing(a.tier, S[0])
    if "splits" in parts:
        for k in ks:
            if str(k) in a.splits.split(","):
                part_split(a.tier, S[k])
    if "nuclear" in parts:
        part_nuclear(a.tier, {k: S[k] for k in ks})
    if "stab" in parts:
        part_stab(a.tier, S[0])
    if "ablate" in parts:
        part_ablate(a.tier, S[0], only=a.only.split(",") if a.only else None)
    _log(f"done {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
