"""torch_ws: GPU working-set Lasso (celer-style outer loop, batched over targets x folds x path).

Per alpha on the path, for all problems b at once:
  1. full check: R = Y_b - X_b w_b for every active problem, C = X^T R in feature chunks (the only
     O(n p B) step). Gives max_j |c_j| (residual-rescaled dual point, hence the duality gap), the
     top-M |c_j| (working-set candidates), and strong-rule counts for the next alpha.
  2. problems with rel gap <= tol are done; others get a working set = support U top candidates,
     sized max(min_ws, 2|support|, strong-rule count) on the first outer iteration of an alpha and
     doubled on each further one.
  3. restricted problems are solved by a CUDA kernel (one thread block per problem, Gram CD) reading
     Gram entries from a union Gram over all features ever put in any working set
     (G_f = X_U^T diag(mask_f) X_U, grown incrementally), to a restricted gap of inner_ratio * tol.
  4. back to 1 for the problems still active.
Warm starts down the path; the final check of alpha a is the first check of alpha a+1.

CV folds are extra problems that share X: fold f = row mask m_f, its own union Gram (downdate is
just a masked matmul on the incremental columns), residuals zeroed off-fold. `fit_cv_batched`
solves all K folds + the full refit path in one pass.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from lassobench.api import PathResult, Solver
from lassobench.solvers._ws_common import (HostX, Timer, cd2_gram_batched, cd_gram_batched, full_corr_stats, gap_from_stats,
                                           max_k_smem, to_csc_list)


class _Union:
    """Union of features ever used in a working set; holds X_U rows, per-fold Grams, X_U^T Y."""

    def __init__(self, Xd, Ym, masks, cap=1024, dtype=torch.float64, gram_budget=None):
        self.Xd, self.Ym, self.masks = Xd, Ym.to(dtype), masks.to(dtype)
        self.dt = dtype
        # union Grams are kept while F * U^2 fits the budget; beyond it, per-problem Grams (gram mode "problem")
        self.gram_budget = gram_budget if gram_budget is not None else 0.3 * torch.cuda.get_device_properties(Ym.device).total_memory
        self.use_gram = True
        self.dev = Ym.device
        self.F = masks.shape[0]
        self.n, self.p = Xd.shape
        self.B = Ym.shape[1]
        self.size = 0
        self.cap = 0
        self.pos_of = torch.full((self.p,), -1, dtype=torch.int64, device=self.dev)
        self.ids = torch.zeros(0, dtype=torch.int64, device=self.dev)
        self.XU = torch.zeros(0, self.n, dtype=dtype, device=self.dev)
        self.G = torch.zeros(self.F, 0, 0, dtype=dtype, device=self.dev)
        self.C0 = torch.zeros(0, self.B, dtype=torch.float64, device=self.dev)
        self.WU = torch.zeros(0, self.B, dtype=torch.float64, device=self.dev)
        self._grow(cap)

    def _grow(self, need):
        if need <= self.cap:
            return
        cap = max(need, int(self.cap * 1.5), 256)
        s = self.size

        esz = torch.finfo(self.dt).bits // 8
        if self.use_gram:
            cap_g = int((self.gram_budget / (self.F * esz)) ** 0.5)
            if need > cap_g:
                self.use_gram = False
                self.G = None
            else:
                cap = min(cap, cap_g)
        XU = torch.zeros(cap, self.n, dtype=self.dt, device=self.dev)
        XU[:s] = self.XU[:s]
        if self.use_gram:
            G = torch.zeros(self.F, cap, cap, dtype=self.dt, device=self.dev)
            G[:, :s, :s] = self.G[:, :s, :s]
        else:
            G = None
        C0 = torch.zeros(cap, self.B, dtype=torch.float64, device=self.dev)
        C0[:s] = self.C0[:s]
        WU = torch.zeros(cap, self.B, dtype=torch.float64, device=self.dev)
        WU[:s] = self.WU[:s]
        ids = torch.zeros(cap, dtype=torch.int64, device=self.dev)
        ids[:s] = self.ids[:s]
        self.XU, self.G, self.C0, self.WU, self.ids = XU, G, C0, WU, ids
        self.cap = cap

    def add(self, gids):
        """Ensure global feature ids `gids` (1-D) are in the union."""
        gids = torch.unique(gids)
        new = gids[self.pos_of[gids] < 0]
        m = new.numel()
        if m == 0:
            return 0
        u0, u1 = self.size, self.size + m
        self._grow(u1)
        self.ids[u0:u1] = new
        self.pos_of[new] = torch.arange(u0, u1, device=self.dev)
        Xn = (self.Xd.cols(new, self.dev) if isinstance(self.Xd, HostX) else self.Xd[:, new]).T.to(self.dt).contiguous()
        self.XU[u0:u1] = Xn
        for f in range(self.F if self.use_gram else 0):
            blk = (self.XU[:u1] * self.masks[f]) @ Xn.T     # (u1, m)
            self.G[f, :u1, u0:u1] = blk
            self.G[f, u0:u1, :u0] = blk[:u0].T
        self.C0[u0:u1] = (Xn @ self.Ym).double()
        self.size = u1
        return m


def _restricted_gap(c, q, w, lam, yy):
    cw = (c * w).sum(1)
    qw = (q * w).sum(1)
    l1 = w.abs().sum(1)
    qm = q.abs().max(1).values
    rr, ry = yy - cw - qw, yy - cw
    return gap_from_stats(rr, ry, l1, qm, lam, yy)


def _polish(un, fa, pos, valid, cmat, w, lam, yy, chunk_bytes=2 ** 30):
    """Active-set Newton step on the current sign pattern, batched (in place on w).

    For A = supp(w), s = sign(w): x_A = G_AA^{-1} (c_A - lam s_A). Move from w toward x up to the first
    sign change (the objective restricted to the orthant is a quadratic minimized at x, so every such
    step decreases it). If no sign flips, this is the exact restricted solution on A.
    Returns the restricted relative duality gap after the step.
    """
    B, K = w.shape
    gap = torch.empty(B, dtype=torch.float64, device=w.device)
    bc = max(1, int(chunk_bytes // (K * K * 8 * 4)))
    eye = torch.eye(K, dtype=torch.float64, device=w.device)
    for i0 in range(0, B, bc):
        sl = slice(i0, min(B, i0 + bc))
        v = valid[sl]
        G = un.G[fa[sl, None, None], pos[sl, :, None], pos[sl, None, :]].double()
        G = G * (v[:, :, None] & v[:, None, :])
        wb, cb, lb = w[sl], cmat[sl], lam[sl]
        Am = wb != 0
        sg = torch.sign(wb)
        Gm = G * (Am[:, :, None] & Am[:, None, :]) + eye * (~Am)[:, None, :]
        rhs = (cb - lb[:, None] * sg) * Am
        L, info = torch.linalg.cholesky_ex(Gm)
        x = torch.cholesky_solve(rhs[..., None], L)[..., 0]
        ok = (info == 0) & torch.isfinite(x).all(1)
        flip = Am & (x * sg <= 0)
        tj = torch.where(flip, wb / (wb - x), torch.full_like(wb, float("inf")))
        t = torch.clamp(tj.min(1).values, max=1.0)
        t = torch.where(ok, t, torch.zeros_like(t))
        wn = wb + t[:, None] * torch.nan_to_num(x - wb)
        wn = torch.where(flip & (tj <= t[:, None] * (1 + 1e-12)), torch.zeros_like(wn), wn)
        wn = wn * Am
        q = cb - (G @ wn[..., None])[..., 0]
        g_new = _restricted_gap(cb, q * v, wn, lb, yy[sl])
        w[sl] = wn
        gap[sl] = g_new
    return gap


def _problem_grams(un, fa, pos, valid, chunk_bytes=2 ** 31, dtype=None):
    """Per-problem Grams G_b = X_Sb^T diag(m_f) X_Sb, (B, K, K), chunked over problems."""
    B, K = pos.shape
    esz = torch.finfo(un.dt).bits // 8
    G = torch.empty(B, K, K, dtype=dtype or un.dt, device=pos.device)
    bc = max(1, int(chunk_bytes // (K * un.n * esz)))
    for i0 in range(0, B, bc):
        sl = slice(i0, min(B, i0 + bc))
        Xs = un.XU[pos[sl]] * valid[sl, :, None]                     # (b, K, n)
        G[sl] = torch.bmm(Xs * un.masks[fa[sl]][:, None, :], Xs.transpose(1, 2)).to(G.dtype)
        del Xs
    idx = torch.arange(K, device=pos.device)
    G[:, idx, idx] = torch.where(valid, G[:, idx, idx], torch.ones_like(G[:, idx, idx]))
    goff = torch.arange(B, device=pos.device) * (K * K)
    return G.view(-1), goff, K, idx[None, :].expand(B, K)


def _inner_solve(un, gview, fa, pos, valid, kk, cmat, dg, w, lam, yy, tol_in, max_ep, act_passes, polish,
                 cd_epochs0, tm, anderson=True, kernel=2, act_passes2=25, snap_every=5):
    """Restricted solves: alternate batched Newton polish and the Gram-CD kernel."""
    B = w.shape[0]
    Gflat, goff, ldg, posk = gview
    pos32 = posk.to(torch.int32)
    kk32 = kk.to(torch.int32)
    ep_tot = torch.zeros(B, dtype=torch.int64, device=w.device)
    gap = torch.full((B,), float("inf"), dtype=torch.float64, device=w.device)
    sub = torch.arange(B, device=w.device)
    npol = 0
    for r in range(polish + 1):
        last = r == polish
        if r > 0 or polish == 0:
            e = max_ep if last else cd_epochs0 * (2 ** (r - 1) if r > 0 else 1)
            with tm("inner_cd"):
                ws = w[sub].contiguous()
                ex = dict(snap_every=snap_every) if kernel == 2 else {}
                fn = cd2_gram_batched if kernel == 2 else cd_gram_batched
                ep, g = fn(Gflat, goff[sub].contiguous(), ldg, pos32[sub].contiguous(),
                                        kk32[sub].contiguous(), cmat[sub].contiguous(), dg[sub].contiguous(),
                                        ws, lam[sub].contiguous(), yy[sub].contiguous(),
                                        tol_in[sub].contiguous(), max_ep=e,
                                        act_passes=act_passes if kernel == 1 else act_passes2,
                                        anderson=anderson, **ex)
                w[sub] = ws
                ep_tot[sub] += ep.long()
                gap[sub] = g
            sub = sub[g > tol_in[sub]]
            if sub.numel() == 0 or last:
                break
        if polish > 0 and un.use_gram:
            with tm("inner_polish"):
                ws = w[sub].contiguous()
                g = _polish(un, fa[sub], pos[sub], valid[sub], cmat[sub], ws, lam[sub], yy[sub])
                w[sub] = ws
                gap[sub] = g
                npol += 1
            sub = sub[g > tol_in[sub]]
            if sub.numel() == 0:
                break
    return w, ep_tot, gap, npol


def _solve_batched(X, Y, alphas, masks, tol, max_outer=100, min_ws=64, inner_ratio=0.3,
                   max_ep=20000, act_passes=4, device="cuda", verbose=False, chunk_bytes=2 ** 31,
                   gram_dtype="float64", polish=0, cd_epochs0=20, anderson=True, kernel=2,
                   act_passes2=25, snap_every=5, stop_after=None, extract_at=None, val_mse=False,
                   tol_margin=0.9, gram_budget=None, ws_rule="celer", problem_gram_dtype="auto",
                   stream_x=False):
    """Core solver. X (n, p) numpy or CUDA tensor, Y (n, T) numpy; alphas (A, T); masks (F, n) {0,1}.

    Problems b = f * T + t. Returns per-alpha triplets (feature, problem, value) and info.
    stop_after: (B,) last alpha index to solve per problem (default all).
    extract_at: (B,) if given, only problem b's solution at alpha extract_at[b] is returned.
    val_mse: also return (A, B) held-out MSE on rows where mask == 0 (info["val_mse"]).
    """
    tm = Timer()
    t_start = time.perf_counter()
    tol_user = tol
    tol = tol * tol_margin       # fp32 full-correlation pass: keep a margin below the requested gap
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device)
    with tm("h2d"):
        if torch.is_tensor(X) or isinstance(X, HostX):
            Xd = X
        elif stream_x:
            Xd = HostX(X)
        else:
            Xd = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
        Yd = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(dev)
        Md = torch.as_tensor(masks, dtype=torch.float32, device=dev)
    n, p = Xd.shape
    T = Yd.shape[1]
    F = Md.shape[0]
    B = F * T
    A = alphas.shape[0]
    fold_of = torch.arange(B, device=dev) // T
    tgt_of = torch.arange(B, device=dev) % T
    with tm("prepare"):
        Ym = (Yd[:, tgt_of] * Md[fold_of].T).contiguous()      # (n, B) masked targets
        yy = (Ym.double() ** 2).sum(0)
        nb = Md.double().sum(1)[fold_of]
        lam_all = torch.as_tensor(np.asarray(alphas, dtype=np.float64), device=dev)[:, tgt_of] * nb[None, :]
        un = _Union(Xd, Ym, Md, dtype=getattr(torch, gram_dtype), gram_budget=gram_budget)
        Kcap = max_k_smem()
    stats = dict(rr=yy.clone(), ry=yy.clone(), l1=torch.zeros(B, dtype=torch.float64, device=dev),
                 qmax=torch.zeros(B, dtype=torch.float64, device=dev),
                 top_idx=None, n_strong=torch.zeros(B, dtype=torch.int64, device=dev))
    ws_size = torch.full((B,), min_ws, dtype=torch.int64, device=dev)
    stop_t = torch.as_tensor(np.asarray(stop_after), device=dev).long() if stop_after is not None else \
        torch.full((B,), A, dtype=torch.long, device=dev)
    ext_t = torch.as_tensor(np.asarray(extract_at), device=dev).long() if extract_at is not None else None
    if val_mse:
        vmask = (1 - Md[fold_of]).double()                       # (B, n)
        nval = vmask.sum(1).clamp(min=1)
        Yfull = Yd[:, tgt_of].T.double()                          # (B, n) unmasked targets
        mse = torch.zeros(A, B, dtype=torch.float64, device=dev)

    def check(act, thr, M):
        """Residuals + full correlation stats for problems `act` (1-D long tensor)."""
        s = un.size
        with tm("residual"):
            Wa = un.WU[:s, act].T.to(un.dt)                         # (|A|, s)
            Ya = un.Ym[:, act].T
            R64 = Ya - (Wa @ un.XU[:s]) * un.masks[fold_of[act]]
            rr = (R64.double() ** 2).sum(1)
            ry = (R64.double() * Ya.double()).sum(1)
            Rt = R64.float()
            l1 = un.WU[:s, act].abs().sum(0)
        with tm("full_corr"):
            st = full_corr_stats(Xd, Rt, M, thr=thr, chunk_bytes=chunk_bytes)
        return rr, ry, l1, st

    # initial check at w = 0
    all_b = torch.arange(B, device=dev)
    thr0 = 2 * lam_all[0] - lam_all[0]
    M = min(p, max(4 * min_ws, 256))
    rr, ry, l1, st = check(all_b, thr0, M)
    stats["rr"], stats["ry"], stats["l1"], stats["qmax"] = rr, ry, l1, st["qmax"]
    top_idx = torch.full((B, M), -1, dtype=torch.int64, device=dev)
    top_idx[:, :st["top_idx"].shape[1]] = st["top_idx"]
    n_strong = st["n_strong"]
    n_matmul = 1
    rows, cols, vals = [], [], []
    info = dict(per_alpha=[])
    for a in range(A):
        lam = lam_all[a]
        nnz = (un.WU[:un.size] != 0).sum(0)
        mw = torch.full_like(nnz, min_ws)
        if ws_rule == "strong":     # strong-rule set, at least 2|supp|
            ws0 = torch.maximum(torch.maximum(2 * nnz, n_strong + 8), mw)
        else:                       # celer-like 2|supp|, shrunk to the strong set when that is smaller
            ws0 = torch.maximum(torch.minimum(torch.maximum(2 * nnz, mw), n_strong + 16), nnz + 16)
        ws_size = torch.clamp(ws0, max=min(Kcap, p))
        gap = gap_from_stats(stats["rr"], stats["ry"], stats["l1"], stats["qmax"], lam, yy)
        act = all_b[(gap > tol) & (stop_t >= a)]
        it = 0
        rec = dict(outer=0, ws_max=0, ws_mean=0.0, epochs_max=0, union=0, nnz_max=0)
        while act.numel() > 0 and it < max_outer:
            it += 1
            with tm("build_ws"):
                Aa = act.numel()
                s = un.size
                Wabs = un.WU[:s, act].abs()                                 # (s, |A|)
                nz = Wabs > 0
                S = int(nz.sum(0).max().item()) if s > 0 else 0
                parts_id, parts_ok = [], []
                if S > 0:
                    v, ppos = torch.topk(nz.float(), S, dim=0)              # (S, |A|)
                    parts_id.append(un.ids[ppos].T)
                    parts_ok.append(v.T > 0)
                cand = top_idx[act]                                          # (|A|, M)
                okc = cand >= 0
                cpos = un.pos_of[cand.clamp(min=0)]
                in_supp = (cpos >= 0) & okc
                if S > 0:
                    insv = un.WU[cpos.clamp(min=0), act[:, None]] != 0
                    in_supp = in_supp & insv
                else:
                    in_supp = torch.zeros_like(okc)
                parts_id.append(cand.clamp(min=0))
                parts_ok.append(okc & ~in_supp)
                ids = torch.cat(parts_id, 1)
                ok = torch.cat(parts_ok, 1)
                wsz = ws_size[act]
                keep = ok & (torch.cumsum(ok.long(), 1) <= wsz[:, None])
                kk = keep.sum(1)
                K = int(kk.max().item())
                order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :K]
                ids = torch.gather(ids, 1, order)
                valid = torch.gather(keep, 1, order)
            with tm("union_gram"):
                un.add(ids[valid])
            with tm("gather"):
                pos = un.pos_of[ids]
                pos = torch.where(valid, pos, torch.full_like(pos, 2 ** 62))
                pos, _ = torch.sort(pos, dim=1)
                valid = pos < 2 ** 62
                pos = torch.where(valid, pos, torch.zeros_like(pos))
                fa = fold_of[act]
                cmat = un.C0[pos, act[:, None]] * valid
                if un.use_gram:
                    gview = (un.G, fa * (un.cap * un.cap), un.cap, pos)
                    dg = un.G[fa[:, None], pos, pos].double()
                else:
                    with tm("problem_gram"):
                        pgd = problem_gram_dtype if problem_gram_dtype != "auto" else \
                            ("float32" if tol >= 1e-5 else "float64")
                        gview = _problem_grams(un, fa, pos, valid, dtype=getattr(torch, pgd))
                    dg = torch.diagonal(gview[0].view(Aa, K, K), dim1=1, dim2=2).double()
                    rec["gram_mode"] = "problem"
                cmat = cmat.double()
                dg = torch.where(valid, dg, torch.ones_like(dg))
                w0 = un.WU[pos, act[:, None]] * valid
                tol_in = torch.full((Aa,), tol * inner_ratio, dtype=torch.float64, device=dev)
            with tm("inner"):
                wout, ep, gin, npol = _inner_solve(un, gview, fa, pos, valid, kk, cmat, dg, w0, lam[act], yy[act],
                                                   tol_in, max_ep, act_passes, polish, cd_epochs0, tm, anderson, kernel,
                                                   act_passes2, snap_every)
            with tm("scatter"):
                un.WU[:, act] = 0
                bb = act[:, None].expand_as(pos)
                un.WU[pos[valid], bb[valid]] = wout[valid]
            rec["ws_max"] = max(rec["ws_max"], K)
            rec["ws_mean"] = max(rec["ws_mean"], float(kk.double().mean()))
            rec["epochs_max"] = max(rec["epochs_max"], int(ep.max()))
            thr = (2 * lam_all[a + 1] - lam)[act] if a + 1 < A else None
            nnz_a = (un.WU[:un.size, act] != 0).sum(0)
            M = int(min(p, Kcap, max(256, 2 * int(ws_size[act].max()), 2 * int(nnz_a.max()) + min_ws)))
            rr, ry, l1, st = check(act, thr, M)
            n_matmul += 1
            stats["rr"][act], stats["ry"][act], stats["l1"][act], stats["qmax"][act] = rr, ry, l1, st["qmax"]
            if top_idx.shape[1] < M:
                ti = torch.full((B, M), -1, dtype=torch.int64, device=dev)
                ti[:, :top_idx.shape[1]] = top_idx
                top_idx = ti
            top_idx[act] = -1
            top_idx[act, :st["top_idx"].shape[1]] = st["top_idx"]
            if thr is not None:
                n_strong[act] = st["n_strong"]
            g = gap_from_stats(rr, ry, l1, st["qmax"], lam[act], yy[act])
            gap[act] = g
            ws_size[act] = torch.clamp(2 * ws_size[act], max=min(Kcap, p))
            act = act[g > tol]
        if act.numel() > 0 and a + 1 < A:
            # problems that hit max_outer: still need strong counts for the next alpha
            pass
        s = un.size
        nzm = un.WU[:s] != 0
        with tm("extract"):
            if ext_t is None:
                r_, c_ = nzm.nonzero(as_tuple=True)
            else:
                r_, c_ = (nzm & (ext_t == a)[None, :]).nonzero(as_tuple=True)
            rows.append(un.ids[r_])
            cols.append(c_)
            vals.append(un.WU[r_, c_])
        if val_mse:
            with tm("val_mse"):
                nzr = nzm.any(1).nonzero().ravel()
                Rf = Yfull - un.WU[nzr].T.double() @ un.XU[nzr].double()
                mse[a] = ((Rf * Rf) * vmask).sum(1) / nval
        rec.update(outer=it, union=s, nnz_max=int(nzm.sum(0).max()) if s else 0,
                   gap_max=float(gap.max()), n_unconv=int(act.numel()))
        info["per_alpha"].append(rec)
        if verbose:
            print(f"a={a:2d} outer={it} ws_max={rec['ws_max']} union={s} nnz_max={rec['nnz_max']} "
                  f"ep_max={rec['epochs_max']} gap_max={rec['gap_max']:.2e}", flush=True)
    with tm("d2h"):
        rows = [r.cpu().numpy() for r in rows]
        cols = [c.cpu().numpy() for c in cols]
        vals = [v.cpu().numpy() for v in vals]
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    if val_mse:
        info["val_mse"] = mse.cpu().numpy()
    info.update(timings=tm.t, n_full_matmul=n_matmul, union_final=un.size, gram_mode="union" if un.use_gram else "problem",
                time_wall=time.perf_counter() - t_start, gpu=torch.cuda.get_device_name(dev),
                peak_gpu_mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9, Kcap=Kcap)
    return rows, cols, vals, info


class TorchWS(Solver):
    name = "torch_ws"
    device = "cuda"

    def __init__(self, tol=1e-4, max_iter=10000, **kw):
        super().__init__(tol=tol, max_iter=max_iter, **kw)

    def fit_path(self, X, Y, alphas):
        n, p = X.shape
        T = Y.shape[1]
        masks = np.ones((1, n), dtype=np.float32)
        rows, cols, vals, info = _solve_batched(X, Y, np.asarray(alphas), masks, self.tol,
                                                max_ep=self.max_iter, **self.kw)
        coefs = to_csc_list(rows, cols, vals, len(rows), p, T)
        return PathResult(coefs=coefs, info=info)

    def fit_cv(self, X, Y, alphas, folds):
        """All K folds as row-masked problems sharing X (and its GPU copy), then a refit on all rows in
        which each target stops at its own selected alpha. Same selection rule as the harness default."""
        import scipy.sparse as sp
        t0 = time.perf_counter()
        n, p = X.shape
        T = Y.shape[1]
        A = alphas.shape[0]
        K = len(folds)
        kw = dict(self.kw)
        cv_batch = kw.pop("cv_batch", "auto")
        Xd = HostX(X) if kw.get("stream_x") else torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        t_h2d = time.perf_counter() - t0
        masks = np.zeros((K, n), dtype=np.float32)
        for k, (tr, va) in enumerate(folds):
            masks[k, tr] = 1
        if cv_batch == "auto":
            # all folds in one batch: union Grams fall back to per-problem Grams when they do not fit
            cv_batch = K
        mse = np.zeros((K, A, T))
        infos = []
        for g0 in range(0, K, cv_batch):
            g = list(range(g0, min(K, g0 + cv_batch)))
            _, _, _, inf = _solve_batched(Xd, Y, np.asarray(alphas), masks[g], self.tol, max_ep=self.max_iter,
                                          extract_at=np.full(len(g) * T, -1), val_mse=True, **kw)
            mse[g] = inf.pop("val_mse").reshape(A, len(g), T).transpose(1, 0, 2)
            infos.append({k: v for k, v in inf.items() if k != "per_alpha"})
            torch.cuda.empty_cache()
        cv_mse = mse.mean(0)
        best = np.argmin(cv_mse, axis=0)
        amax = int(best.max())
        rows, cols, vals, inf = _solve_batched(Xd, Y, np.ascontiguousarray(alphas[:amax + 1]), np.ones((1, n), np.float32),
                                               self.tol, max_ep=self.max_iter, stop_after=best, extract_at=best, **kw)
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        coef = sp.csc_matrix((v, (r, c)), shape=(p, T))
        del Xd
        torch.cuda.empty_cache()
        info = dict(fold_infos=infos, refit_info={k: v for k, v in inf.items() if k != "per_alpha"},
                    cv_batch=cv_batch, time_h2d_s=t_h2d, time_total_inner_s=time.perf_counter() - t0)
        return dict(coef=coef, best_idx=best, cv_mse=cv_mse, info=info)


SOLVERS = {"torch_ws": TorchWS}
