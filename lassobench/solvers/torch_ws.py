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
from lassobench.solvers._ws_common import (HostX, Timer, warm_kernels, cd2_gram_batched, cd_gram_batched, full_corr_stats, gap_from_stats,
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


def _fista_restricted(G, c, w, lam, yy, tol_in, max_it, check_every=10):
    """Batched FISTA (gradient restart) on restricted problems 0.5 w'Gw - c'w + lam|w|_1.

    G (b, K, K), c/w (b, K) in G's dtype. All problems advance together with one bmm per iteration;
    converged problems are compacted out at each check. Returns (w, iters (b,), restricted rel gap (b,))."""
    b, K = w.shape
    dt = G.dtype
    dev = w.device
    # Lipschitz constants by power iteration
    v = torch.randn(b, K, 1, dtype=dt, device=dev)
    for _ in range(25):
        v = torch.bmm(G, v)
        v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp(min=1e-30)
    L = (torch.bmm(G, v) * v).sum((1, 2)).clamp(min=1e-30) * 1.05
    lamd, cd = lam.to(dt), c.to(dt)
    x = w.to(dt).clone()
    out_w = w.clone()
    out_it = torch.zeros(b, dtype=torch.int64, device=dev)
    out_gap = torch.full((b,), float("inf"), dtype=torch.float64, device=dev)
    idx = torch.arange(b, device=dev)
    y, xp = x.clone(), x.clone()
    t = torch.ones(b, dtype=dt, device=dev)
    it = 0
    while idx.numel() > 0 and it < max_it:
        Gy = torch.bmm(G, y[:, :, None])[:, :, 0]
        z = y - (Gy - cd) / L[:, None]
        th = (lamd / L)[:, None]
        xn = torch.sign(z) * torch.clamp(z.abs() - th, min=0)
        rst = ((y - xn) * (xn - x)).sum(1) > 0
        tn = torch.where(rst, torch.ones_like(t), (1 + torch.sqrt(1 + 4 * t * t)) / 2)
        beta = torch.where(rst, torch.zeros_like(t), (t - 1) / tn)
        y = xn + beta[:, None] * (xn - x)
        x, t = xn, tn
        it += 1
        if it % check_every == 0 or it == max_it:
            q = cd - torch.bmm(G, x[:, :, None])[:, :, 0]
            g = _restricted_gap(cd.double(), q.double(), x.double(), lam.double(), yy)
            done = (g <= tol_in) | (it >= max_it)
            if done.any():
                di = idx[done]
                out_w[di] = x[done].to(out_w.dtype)
                out_it[di] = it
                out_gap[di] = g[done]
                keep = ~done
                idx, G, cd, lamd, lam, yy, tol_in, L = idx[keep], G[keep], cd[keep], lamd[keep], lam[keep], yy[keep], tol_in[keep], L[keep]
                x, y, t = x[keep], y[keep], t[keep]
    return out_w, out_it, out_gap


def _relaxed_refit(un, fa, gamma, chunk_bytes=2 ** 32):
    """Ridge refit on each problem's current support (relaxed Lasso), batched on the GPU.

    For problem b with support S (nonzeros of WU[:, b]): beta_S = (G_SS + gamma * mean(diag G_SS) I)^{-1} c_S,
    with G_SS from the union Gram when it is held, else from X_U rows (fold-masked). Returns a dense
    (size, B) coefficient matrix over the union (zero off-support)."""
    s = un.size
    W = un.WU[:s]
    nz = W != 0
    kk = nz.sum(0)
    B = W.shape[1]
    out = torch.zeros_like(W)
    k = int(kk.max()) if B else 0
    if k == 0:
        return out
    v, pos = torch.topk(nz.double(), k, dim=0)               # (k, B) positions of the support (padded)
    pos, valid = pos.T.contiguous(), (v.T > 0)
    bc = max(1, int(chunk_bytes // (k * max(k, un.n) * 8)))
    ar = torch.arange(k, device=W.device)
    for i0 in range(0, B, bc):
        sl = slice(i0, min(B, i0 + bc))
        ps, vl, f = pos[sl], valid[sl], fa[sl]
        if un.use_gram:
            G = un.G[f[:, None, None], ps[:, :, None], ps[:, None, :]].double()
        else:
            Xs = un.XU[ps].double()
            G = torch.bmm(Xs * un.masks[f][:, None, :].double(), Xs.transpose(1, 2))
            del Xs
        m = vl[:, :, None] & vl[:, None, :]
        G = G * m
        dgm = (torch.diagonal(G, dim1=1, dim2=2) * vl).sum(1) / vl.sum(1).clamp(min=1)
        G[:, ar, ar] += torch.where(vl, gamma * dgm[:, None], torch.ones_like(G[:, ar, ar]))
        bidx = torch.arange(sl.start, sl.stop, device=W.device)
        c = un.C0[ps, bidx[:, None]] * vl
        L, info = torch.linalg.cholesky_ex(G)
        beta = torch.cholesky_solve(c[..., None], L)[..., 0] * vl
        beta = torch.where((info == 0)[:, None], beta, W[ps, bidx[:, None]] * vl)   # fall back to the Lasso coef
        out[ps[vl], bidx[:, None].expand_as(ps)[vl]] = beta[vl]
    return out


def _gather_grams(un, gview, fa, pos, valid, dtype):
    """(b, K, K) dense restricted Grams (padding: identity) from the union Gram or the per-problem view."""
    B, K = pos.shape
    if un.use_gram:
        G = un.G[fa[:, None, None], pos[:, :, None], pos[:, None, :]].to(dtype)
    else:
        G = gview[0].view(B, K, K).to(dtype)
    m = valid[:, :, None] & valid[:, None, :]
    G = G * m
    ar = torch.arange(K, device=pos.device)
    G[:, ar, ar] = torch.where(valid, G[:, ar, ar], torch.ones_like(G[:, ar, ar]))
    return G


def _inner_solve(un, gview, fa, pos, valid, kk, cmat, dg, w, lam, yy, tol_in, max_ep, act_passes, polish,
                 cd_epochs0, tm, anderson=True, kernel=2, act_passes2=25, snap_every=5, inner="cd",
                 fista_iters=3000, fista_dtype=torch.float32, fista_bytes=2 ** 33, l2=None):
    """Restricted solves: batched FISTA on dense restricted Grams (inner="fista", falls back to the CD
    kernel for problems it leaves unconverged) or the Gram-CD kernel (inner="cd"), optional Newton polish."""
    B = w.shape[0]
    if inner == "fista":
        K = w.shape[1]
        esz = torch.finfo(fista_dtype).bits // 8
        bc = max(1, int(fista_bytes // (K * K * esz)))
        gap = torch.empty(B, dtype=torch.float64, device=w.device)
        its = torch.zeros(B, dtype=torch.int64, device=w.device)
        with tm("inner_fista"):
            for i0 in range(0, B, bc):
                sl = slice(i0, min(B, i0 + bc))
                G = _gather_grams(un, gview, fa[sl], pos[sl], valid[sl], fista_dtype)
                if l2 is not None:
                    ar = torch.arange(K, device=w.device)
                    G[:, ar, ar] += (l2[sl, None] * valid[sl]).to(G.dtype)
                wn, it_, g = _fista_restricted(G, cmat[sl] * valid[sl], w[sl], lam[sl], yy[sl], tol_in[sl], fista_iters)
                del G
                w[sl] = wn * valid[sl]
                gap[sl] = g
                its[sl] = it_
        left = gap > tol_in
        if not left.any():
            return w, its, gap, 0
        # finish the stragglers with the CD kernel
        sub_ = left.nonzero().ravel()
        w2, ep2, g2, _ = _inner_solve(un, (gview[0], gview[1][sub_], gview[2], gview[3][sub_]), fa[sub_], pos[sub_],
                                      valid[sub_], kk[sub_], cmat[sub_], dg[sub_], w[sub_].contiguous(), lam[sub_],
                                      yy[sub_], tol_in[sub_], max_ep, act_passes, polish, cd_epochs0, tm, anderson,
                                      kernel, act_passes2, snap_every, inner="cd",
                                      l2=l2[sub_] if l2 is not None else None)
        w[sub_] = w2
        gap[sub_] = g2
        its[sub_] += ep2
        return w, its, gap, 0
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
                ex = dict(snap_every=snap_every, l2=l2[sub].contiguous() if l2 is not None else None) if kernel == 2 else {}
                if kernel != 2 and l2 is not None:
                    raise ValueError("elastic net needs kernel=2")
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
                   stream_x=False, inner="cd", fista_iters=3000, norm_score=True, cv_stop=None, l1_ratio=1.0,
                   relax_gamma=None, relax_extract=True):
    """Core solver. X (n, p) numpy or CUDA tensor, Y (n, T) numpy; alphas (A, T); masks (F, n) {0,1}.

    Problems b = f * T + t. Returns per-alpha triplets (feature, problem, value) and info.
    stop_after: (B,) last alpha index to solve per problem (default all).
    extract_at: (B,) if given, only problem b's solution at alpha extract_at[b] is returned.
    val_mse: also return (A, B) held-out MSE on rows where mask == 0 (info["val_mse"]).
    """
    warm_kernels()
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
        # elastic net as an augmented Lasso: lam1 = n alpha (the given grid), lam2 = lam1 (1 - rho) / rho
        enet = l1_ratio < 1.0
        l2_all = lam_all * ((1.0 - l1_ratio) / l1_ratio) if enet else None
        if isinstance(Xd, HostX):
            cn = torch.cat([(b_.to(dev) ** 2).sum(0) for b_ in Xd.blocks])
        else:
            cn = torch.cat([(Xd[:, j:j + 65536].double() ** 2).sum(0).float() for j in range(0, p, 65536)])
        inv_norm = (1.0 / torch.sqrt(cn).clamp(min=1e-12)) if norm_score else None
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
        mse = torch.full((A, B), float("inf"), dtype=torch.float64, device=dev)
        mse_lasso = torch.full((A, B), float("inf"), dtype=torch.float64, device=dev)
        if cv_stop is not None:
            cv_best = torch.full((T,), float("inf"), dtype=torch.float64, device=dev)
            cv_arg = torch.zeros(T, dtype=torch.long, device=dev)

    def check(act, thr, M, lam_c=None, l2c=None):
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
            corr = None
            if l2c is not None:
                Wsub = un.WU[:s, act]
                rr = rr + l2c * (Wsub * Wsub).sum(0)
                r_, c_ = (Wsub != 0).nonzero(as_tuple=True)
                corr = (un.ids[r_], c_, (l2c[c_] * Wsub[r_, c_]).float())
        with tm("full_corr"):
            st = full_corr_stats(Xd, Rt, M, thr=thr, chunk_bytes=chunk_bytes,
                                 inv_norm=inv_norm if lam_c is not None else None, lam=lam_c, corr=corr)
        return rr, ry, l1, st

    # initial check at w = 0
    all_b = torch.arange(B, device=dev)
    thr0 = 2 * lam_all[0] - lam_all[0]
    M = min(p, max(4 * min_ws, 256))
    rr, ry, l1, st = check(all_b, thr0, M, lam_all[0])
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
        # elastic net: the stats of the previous check used the previous lam2, so always re-solve
        act = all_b[((gap > tol) | enet) & (stop_t >= a)]
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
                if enet:
                    dg = dg + l2_all[a][act][:, None]
                dg = torch.where(valid, dg, torch.ones_like(dg))
                w0 = un.WU[pos, act[:, None]] * valid
                tol_in = torch.full((Aa,), tol * inner_ratio, dtype=torch.float64, device=dev)
            with tm("inner"):
                wout, ep, gin, npol = _inner_solve(un, gview, fa, pos, valid, kk, cmat, dg, w0, lam[act], yy[act],
                                                   tol_in, max_ep, act_passes, polish, cd_epochs0, tm, anderson, kernel,
                                                   act_passes2, snap_every, inner, fista_iters,
                                                   getattr(torch, gram_dtype if tol < 1e-5 else "float32"),
                                                   l2=l2_all[a][act] if enet else None)
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
            rr, ry, l1, st = check(act, thr, M, lam[act], l2_all[a][act] if enet else None)
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
        Wrel = None
        if relax_gamma is not None:
            with tm("relax"):
                Wrel = _relaxed_refit(un, fold_of, relax_gamma)
        with tm("extract"):
            if ext_t is None:
                r_, c_ = nzm.nonzero(as_tuple=True)
            else:
                r_, c_ = (nzm & (ext_t == a)[None, :]).nonzero(as_tuple=True)
            rows.append(un.ids[r_])
            cols.append(c_)
            vals.append((Wrel if (Wrel is not None and relax_extract) else un.WU)[r_, c_])
        if val_mse:
            with tm("val_mse"):
                nzr = nzm.any(1).nonzero().ravel()
                Rf = Yfull - un.WU[nzr].T.double() @ un.XU[nzr].double()
                ma = ((Rf * Rf) * vmask).sum(1) / nval
                if Wrel is not None:
                    Rr = Yfull - Wrel[nzr].T.double() @ un.XU[nzr].double()
                    mr = ((Rr * Rr) * vmask).sum(1) / nval
                    mse_lasso[a] = torch.where(stop_t >= a, ma, torch.full_like(ma, float("inf")))
                    ma = mr                 # CV (and early stop) on the relaxed fit
                mse[a] = torch.where(stop_t >= a, ma, torch.full_like(ma, float("inf")))
                if cv_stop is not None:
                    # CV early stop: mean held-out MSE over the F folds of each target; once it has stayed
                    # above (1 + rise) * running min for `patience` alphas, the target's fold paths stop.
                    patience, rise = cv_stop
                    cvm = mse[a].view(F, T).mean(0)
                    better = cvm < cv_best
                    cv_best = torch.where(better, cvm, cv_best)
                    cv_arg = torch.where(better, torch.full_like(cv_arg, a), cv_arg)
                    stop_tgt = ((a - cv_arg) >= patience) & (cvm > cv_best * (1 + rise))
                    stop_t = torch.where(stop_tgt[tgt_of] & (stop_t > a), torch.full_like(stop_t, a), stop_t)
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
        if relax_gamma is not None:
            info["val_mse_lasso"] = mse_lasso.cpu().numpy()
        info["alpha_stop_mean"] = float(torch.clamp(stop_t, max=A - 1).double().mean())
    info.update(timings=tm.t, n_full_matmul=n_matmul, union_final=un.size, gram_mode="union" if un.use_gram else "problem",
                time_wall=time.perf_counter() - t_start, gpu=torch.cuda.get_device_name(dev),
                peak_gpu_mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9, Kcap=Kcap)
    return rows, cols, vals, info


_CV_ONLY = ("select", "cv_stop", "cv_batch", "relax", "relax_grid")
RELAX_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _xcols(Xd, ids, dev):
    """Columns `ids` (1-D long, on dev) of X as a (n, m) float64 GPU tensor (GPU copy or host-streamed X)."""
    if isinstance(Xd, HostX):
        return Xd.cols(ids, dev).double()
    return Xd[:, ids].double()


def _support_ridge(Xd, Yd, rows_fit, rows_val, ids, valid, grid, chunk_bytes=2 ** 31):
    """Batched ridge refits restricted to per-problem supports.

    ids/valid: (B, k) global feature ids of each problem's support (padded) and its mask; Yd (n, B) targets
    (column b belongs to problem b). For each relative penalty g in `grid`:
        beta = (X_S^T X_S + g * mean(diag X_S^T X_S) I)^{-1} X_S^T y   on rows_fit,
    returns (betas (len(grid), B, k), val_mse (len(grid), B) on rows_val or None)."""
    B, k = ids.shape
    dev = ids.device
    L = len(grid)
    betas = torch.zeros(L, B, k, dtype=torch.float64, device=dev)
    vmse = torch.full((L, B), float("inf"), dtype=torch.float64, device=dev) if rows_val is not None else None
    if k == 0 or B == 0:
        return betas, vmse
    n = int(rows_fit.numel()) + (int(rows_val.numel()) if rows_val is not None else 0)
    bc = max(1, int(chunk_bytes // (8 * k * max(n, k) * 3)))
    gr = torch.as_tensor(grid, dtype=torch.float64, device=dev)
    ar = torch.arange(k, device=dev)
    for i0 in range(0, B, bc):
        sl = slice(i0, min(B, i0 + bc))
        idc, vl = ids[sl], valid[sl]
        u, inv = torch.unique(idc[vl], return_inverse=True)
        if u.numel() == 0:
            continue
        Xu = _xcols(Xd, u, dev)                                     # (n, m)
        loc = torch.zeros_like(idc)
        loc[vl] = inv
        Xf = Xu[rows_fit]                                           # (n_fit, m)
        Xs = Xf[:, loc].permute(1, 0, 2) * vl[:, None, :]           # (b, n_fit, k)
        G = torch.bmm(Xs.transpose(1, 2), Xs)
        c = torch.bmm(Xs.transpose(1, 2), Yd[rows_fit][:, sl].T.double()[:, :, None])[:, :, 0]
        del Xs
        md = (torch.diagonal(G, dim1=1, dim2=2) * vl).sum(1) / vl.sum(1).clamp(min=1)
        if rows_val is not None:
            Xv = Xu[rows_val][:, loc].permute(1, 0, 2) * vl[:, None, :]   # (b, n_val, k)
            yv = Yd[rows_val][:, sl].T.double()                          # (b, n_val)
        for li in range(L):
            Gl = G.clone()
            Gl[:, ar, ar] += torch.where(vl, gr[li] * md[:, None].clamp(min=1e-12), torch.ones_like(Gl[:, ar, ar]))
            Lc, info = torch.linalg.cholesky_ex(Gl)
            bet = torch.cholesky_solve(c[:, :, None], Lc)[:, :, 0] * vl
            bet = torch.where((info == 0)[:, None], bet, torch.zeros_like(bet))
            betas[li, sl] = bet
            if rows_val is not None:
                pred = torch.bmm(Xv, bet[:, :, None])[:, :, 0]
                vmse[li, sl] = ((yv - pred) ** 2).mean(1)
        del G
    return betas, vmse


def _triplets_to_supports(rows, cols, vals, sel_cols, dev):
    """From one alpha's (feature, problem, value) triplets, padded supports for problems `sel_cols` (list)."""
    r = torch.as_tensor(rows, device=dev).long()
    c = torch.as_tensor(cols, device=dev).long()
    v = torch.as_tensor(vals, device=dev).double()
    sel = torch.as_tensor(sel_cols, device=dev).long()
    pos_of = torch.full((int(max(c.max().item() if c.numel() else 0, sel.max().item())) + 1,), -1, dtype=torch.long, device=dev)
    pos_of[sel] = torch.arange(sel.numel(), device=dev)
    keep = pos_of[c] >= 0 if c.numel() else torch.zeros(0, dtype=torch.bool, device=dev)
    r, c, v = r[keep], pos_of[c[keep]], v[keep]
    Bs = sel.numel()
    cnt = torch.bincount(c, minlength=Bs)
    k = int(cnt.max()) if cnt.numel() else 0
    ids = torch.zeros(Bs, max(k, 1), dtype=torch.long, device=dev)
    lv = torch.zeros(Bs, max(k, 1), dtype=torch.float64, device=dev)
    valid = torch.zeros(Bs, max(k, 1), dtype=torch.bool, device=dev)
    if r.numel():
        o = torch.argsort(c, stable=True)
        r, c, v = r[o], c[o], v[o]
        start = torch.cumsum(cnt, 0) - cnt
        slot = torch.arange(r.numel(), device=dev) - start[c]
        ids[c, slot], lv[c, slot], valid[c, slot] = r, v, True
    return ids[:, :max(k, 1)], lv, valid


class TorchWS(Solver):
    name = "torch_ws"
    device = "cuda"

    def __init__(self, tol=1e-4, max_iter=10000, **kw):
        super().__init__(tol=tol, max_iter=max_iter, **kw)

    def fit_path(self, X, Y, alphas):
        n, p = X.shape
        T = Y.shape[1]
        masks = np.ones((1, n), dtype=np.float32)
        kw = {k: v for k, v in self.kw.items() if k not in _CV_ONLY}
        rows, cols, vals, info = _solve_batched(X, Y, np.asarray(alphas), masks, self.tol,
                                                max_ep=self.max_iter, **kw)
        coefs = to_csc_list(rows, cols, vals, len(rows), p, T)
        return PathResult(coefs=coefs, info=info)

    def fit_cv(self, X, Y, alphas, folds):
        """All K folds as row-masked problems sharing X (one GPU copy), then a refit on all rows in which each
        target stops at its selected alpha. Same selection rule as the harness default (first argmin of the
        fold-mean held-out MSE), plus options:
          select="min"|"1se"   1se = largest alpha with cv_mse <= min + SE (SE = fold std / sqrt(K))
          cv_stop=(10, 0.05)   stop a target's fold paths once its CV MSE stayed > 5 % above its running
                               minimum for 10 alphas (None/"none" = full path)
          l1_ratio=rho         elastic net (lam2 = alpha (1 - rho) / rho), relax_gamma=g  relaxed Lasso
                               (ridge refit on each support, CV on the refit error, refit coefs returned)
        Returns also fold_coefs: K sparse (p, T) Lasso coefficient matrices at the selected alpha per target."""
        import scipy.sparse as sp
        t0 = time.perf_counter()
        n, p = X.shape
        T = Y.shape[1]
        A = alphas.shape[0]
        K = len(folds)
        kw = dict(self.kw)
        cv_batch = kw.pop("cv_batch", "auto")
        select = kw.pop("select", "min")
        cv_stop = kw.pop("cv_stop", (10, 0.05))      # None = full path for every fold
        cv_stop = tuple(cv_stop) if (cv_stop is not None and cv_stop != "none") else None
        relax = kw.pop("relax", None)                # "cv": relaxed Lasso, ridge penalty chosen by CV
        relax_grid = tuple(kw.pop("relax_grid", RELAX_GRID))
        Xd = HostX(X) if kw.get("stream_x") else torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
        t_h2d = time.perf_counter() - t0
        masks = np.zeros((K, n), dtype=np.float32)
        for k, (tr, va) in enumerate(folds):
            masks[k, tr] = 1
        if cv_batch == "auto":
            cv_batch = K       # union Grams fall back to per-problem Grams when they do not fit
        mse = np.full((K, A, T), np.inf)
        fold_trip = [None] * K
        infos = []
        for g0 in range(0, K, cv_batch):
            g = list(range(g0, min(K, g0 + cv_batch)))
            rows, cols, vals, inf = _solve_batched(Xd, Y, np.asarray(alphas), masks[g], self.tol, max_ep=self.max_iter,
                                                   val_mse=True, cv_stop=cv_stop if len(g) == K else None,
                                                   relax_extract=False, **kw)
            mse[g] = inf.pop("val_mse").reshape(A, len(g), T).transpose(1, 0, 2)
            inf.pop("val_mse_lasso", None)
            for j, fk in enumerate(g):
                fold_trip[fk] = (rows, cols, vals, j)
            infos.append({k: v for k, v in inf.items() if k != "per_alpha"})
            torch.cuda.empty_cache()
        with np.errstate(invalid="ignore"):
            cv_mse = mse.mean(0)
            se = mse.std(0, ddof=1) / np.sqrt(K)
        best_min = np.argmin(cv_mse, axis=0)
        thr = cv_mse[best_min, np.arange(T)] + se[best_min, np.arange(T)]
        best_1se = np.array([int(np.nonzero(cv_mse[:, t] <= thr[t])[0].min()) for t in range(T)])
        best = best_1se if select == "1se" else best_min
        best_lasso = best
        relax_info = None
        if relax == "cv":
            # relaxed Lasso with the refit ridge penalty chosen by CV: for every fold, alpha and channel, ridge
            # refits on the fold model's support (fold training rows only) over `relax_grid` (relative to the
            # mean support-Gram diagonal) plus the plain Lasso coefficients; (alpha, penalty) chosen jointly.
            tr0 = time.perf_counter()
            dev = torch.device("cuda")
            Yd = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(dev)
            Lg = len(relax_grid)
            mse_r = np.full((K, A, Lg + 1, T), np.inf)
            mse_r[:, :, Lg, :] = mse
            for fk in range(K):
                rows, cols, vals, j = fold_trip[fk]
                tr = torch.as_tensor(folds[fk][0], device=dev).long()
                va = torch.as_tensor(folds[fk][1], device=dev).long()
                for a in range(A):
                    live = np.isfinite(mse[fk, a])
                    if not live.any():
                        continue
                    ids, _, valid = _triplets_to_supports(rows[a], cols[a], vals[a], list(j * T + np.arange(T)), dev)
                    _, vm = _support_ridge(Xd, Yd, tr, va, ids, valid, relax_grid)
                    vm = vm.cpu().numpy()
                    mse_r[fk, a, :Lg, :] = np.where(live[None, :], vm, np.inf)
            with np.errstate(invalid="ignore"):
                cvr = mse_r.mean(0)                                        # (A, L+1, T)
                ser = mse_r.std(0, ddof=1) / np.sqrt(K)
            flat = cvr.reshape(A * (Lg + 1), T)
            bi = np.argmin(flat, axis=0)
            ra_min, rl_min = bi // (Lg + 1), bi % (Lg + 1)
            thr_r = cvr[ra_min, rl_min, np.arange(T)] + ser[ra_min, rl_min, np.arange(T)]
            curve = cvr.min(1)                                             # (A, T) best penalty per alpha
            ra_1se = np.array([int(np.nonzero(curve[:, t] <= thr_r[t])[0].min()) for t in range(T)])
            rl_1se = np.array([int(np.argmin(cvr[ra_1se[t], :, t])) for t in range(T)])
            ra, rl = (ra_1se, rl_1se) if select == "1se" else (ra_min, rl_min)
            relax_info = dict(time_relax_cv_s=time.perf_counter() - tr0, grid=list(relax_grid),
                              alpha_idx=ra.tolist(), penalty_idx=rl.tolist(),
                              frac_no_refit=float(np.mean(rl == Lg)),
                              penalty_idx_per_alpha=np.argmin(np.where(np.isfinite(cvr), cvr, np.inf), axis=1).tolist(),
                              cv_mse_relaxed=np.where(np.isfinite(curve), curve, None).tolist())
            best = ra
        fold_coefs = []
        for fk in range(K):
            rows, cols, vals, j = fold_trip[fk]
            rr_, cc_, vv_ = [], [], []
            for a in range(A):
                m = (cols[a] // T == j) & (best[cols[a] % T] == a)
                rr_.append(rows[a][m]); cc_.append(cols[a][m] % T); vv_.append(vals[a][m])
            fold_coefs.append(sp.csc_matrix((np.concatenate(vv_), (np.concatenate(rr_), np.concatenate(cc_))), shape=(p, T)))
        del fold_trip
        stop = np.maximum(best, best_lasso)
        amax = int(stop.max())
        rows, cols, vals, inf = _solve_batched(Xd, Y, np.ascontiguousarray(alphas[:amax + 1]), np.ones((1, n), np.float32),
                                               self.tol, max_ep=self.max_iter, stop_after=stop,
                                               extract_at=None if relax == "cv" else best, **kw)

        def pick(at):
            rr_, cc_, vv_ = [], [], []
            for a_ in range(amax + 1):
                m = at[cols[a_]] == a_
                rr_.append(rows[a_][m]); cc_.append(cols[a_][m]); vv_.append(vals[a_][m])
            return np.concatenate(rr_), np.concatenate(cc_), np.concatenate(vv_)
        if relax == "cv":
            r_l, c_l, v_l = pick(best_lasso)
            coef_lasso = sp.csc_matrix((v_l, (r_l, c_l)), shape=(p, T))
            tr1 = time.perf_counter()
            r_r, c_r, v_r = pick(best)
            dev = torch.device("cuda")
            ids, lv, valid = _triplets_to_supports(r_r, c_r, v_r, list(range(T)), dev)
            allrows = torch.arange(n, device=dev)
            betas, _ = _support_ridge(Xd, Yd, allrows, None, ids, valid, relax_grid)
            Lg = len(relax_grid)
            rl_t = torch.as_tensor(rl, device=dev).long()
            chosen = betas[rl_t.clamp(max=Lg - 1), torch.arange(T, device=dev)]          # (T, k)
            chosen = torch.where((rl_t == Lg)[:, None], lv, chosen) * valid
            tt = torch.arange(T, device=dev)[:, None].expand_as(ids)
            coef = sp.csc_matrix((chosen[valid].cpu().numpy(), (ids[valid].cpu().numpy(), tt[valid].cpu().numpy())),
                                 shape=(p, T))
            relax_info["time_relax_final_s"] = time.perf_counter() - tr1
            del Yd
        else:
            coef = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(p, T))
            coef_lasso = coef
        del Xd
        torch.cuda.empty_cache()
        info = dict(fold_infos=infos, refit_info={k: v for k, v in inf.items() if k != "per_alpha"},
                    cv_batch=cv_batch, cv_stop=cv_stop, select=select, best_min=best_min.tolist(),
                    best_1se=best_1se.tolist(), time_h2d_s=t_h2d, time_total_inner_s=time.perf_counter() - t0)
        if relax_info is not None:
            info["relax"] = relax_info
            info["best_lasso"] = best_lasso.tolist()
        return dict(coef=coef, best_idx=best, cv_mse=cv_mse, fold_coefs=fold_coefs, info=info, coef_lasso=coef_lasso)


SOLVERS = {"torch_ws": TorchWS}
