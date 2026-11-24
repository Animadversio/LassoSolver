"""Alternating convex fits of the rank-r space x feature readout W_t = sum_r S_{t,r} (x) F_{t,r}.

With S fixed the model is linear in F (design A_t[n, (r,c)] = sum_s S_{t,r,s} x[n,c,s]); with F fixed it is linear
in S (design B_t[n, (r,s)] = sum_c F_{t,r,c} x[n,c,s]). Each block is a Lasso, solved for all T channels at
once by batched FISTA on the GPU (exact zeros from soft-thresholding). Objective per channel:

    1/(2n) ||y_t - <W_t, x>||^2 + a_f,t |F_t|_1 + a_s,t |S_t|_1

which is biconvex. After every S-step each component is rescaled to ||S_{t,r}||_2 = 1 (scale moved into F),
so the penalties act in fixed units; this rescaling means the alternation is a heuristic, not a monotone descent. Init: rank-r SVD of the (C, H*W)-reshaped ridge
weight of each channel. Penalties are set relative to each block's alpha_max at the init (frac_f, frac_s).
"""
from __future__ import annotations

import torch


def soft(x, t):
    return torch.sign(x) * (x.abs() - t).clamp(min=0)


def lipschitz(A, iters=30):
    """Largest eigenvalue of A_t^T A_t / n per batch item. A: (T, n, q)."""
    n = A.shape[1]
    v = torch.randn(A.shape[0], A.shape[2], 1, device=A.device, dtype=A.dtype)
    At = A.transpose(1, 2).contiguous()
    for _ in range(iters):
        v = At @ (A @ v)
        v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return ((A @ v).norm(dim=1) ** 2).squeeze(-1) / n


def fista(A, y, lam, w0, iters=300, L=None, tol=1e-6):
    """Batched Lasso: min 1/(2n)||y_t - A_t w_t||^2 + lam_t |w_t|_1. A (T,n,q), y (T,n), lam (T,), w0 (T,q)."""
    n = A.shape[1]
    L = (lipschitz(A) * 1.01 if L is None else L).clamp(min=1e-12)
    step = (1.0 / L)[:, None]
    w, z = w0.clone(), w0.clone()
    tk = torch.ones(A.shape[0], 1, device=A.device, dtype=A.dtype)
    At = A.transpose(1, 2).contiguous()
    for it in range(iters):
        r = (A @ z[..., None])[..., 0] - y
        g = (At @ r[..., None])[..., 0] / n
        w_new = soft(z - step * g, step * lam[:, None])
        tk1 = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        z = w_new + ((tk - 1) / tk1) * (w_new - w)
        # adaptive restart (O'Donoghue & Candes, gradient scheme) when momentum points uphill
        up = ((w_new - w) * g).sum(1, keepdim=True) > 0
        z = torch.where(up, w_new, z)
        tk1 = torch.where(up, torch.ones_like(tk1), tk1)
        done = False
        if it % 25 == 24:
            dw = (w_new - w).norm(dim=1) / w_new.norm(dim=1).clamp(min=1e-12)
            done = float(dw.max()) < tol
        w, tk = w_new, tk1
        if done:
            break
    return w


def normalize(S, Fv):
    """Rescale each component so ||S_{t,r}||_2 = 1 (scale moved into F): fixes the S/F scale ambiguity that
    otherwise lets the mask grow to undo the feature penalty (effective penalty ~ sqrt(a_f a_s))."""
    nrm = S.norm(dim=2, keepdim=True).clamp(min=1e-12)
    return S / nrm, Fv * nrm


def design_F(Xc, S):
    """Xc (n, C, s), S (T, r, s) -> A (T, n, r*C)."""
    T, r, _ = S.shape
    return torch.einsum("ncs,trs->tnrc", Xc, S).reshape(T, Xc.shape[0], r * Xc.shape[1])


def design_S(Xc, Fv):
    """Xc (n, C, s), Fv (T, r, C) -> B (T, n, r*s)."""
    T, r, _ = Fv.shape
    return torch.einsum("ncs,trc->tnrs", Xc, Fv).reshape(T, Xc.shape[0], r * Xc.shape[2])


def predict(X, S, Fv, b):
    """X (N, C, s) -> (N, T)."""
    return torch.einsum("ncs,trs,trc->nt", X, S, Fv) + b


def ridge_dual(Xf, Y, Xva_f, Yva, lams):
    """Per-channel ridge on flattened X, lambda chosen on validation R2. Returns W (p, T), chosen lambda (T,)."""
    mx, my = Xf.mean(0), Y.mean(0)
    Xc, Yc = Xf - mx, Y - my
    K = Xc @ Xc.T
    ev, U = torch.linalg.eigh(K.double())
    UtY = U.T @ Yc.double()
    best, bestW, bestlam = None, None, None
    for lam in lams:
        alpha = U @ (UtY / (ev + lam)[:, None])                  # (n, T)
        W = (Xc.T.double() @ alpha).float()
        pv = (Xva_f - mx) @ W + my
        r2 = 1 - ((Yva - pv) ** 2).sum(0) / ((Yva - Yva.mean(0)) ** 2).sum(0)
        if best is None:
            best, bestW, bestlam = r2, W, torch.full_like(r2, lam)
        else:
            m = r2 > best
            best = torch.where(m, r2, best)
            bestW = torch.where(m[None], W, bestW)
            bestlam = torch.where(m, torch.full_like(r2, lam), bestlam)
    return bestW, bestlam, best


def svd_init(W, shape, rank):
    """W (p, T) -> S (T, r, s), F (T, r, C) from the rank-r SVD of each channel's (C, s) weight map."""
    C, H, Wd = shape
    M = W.T.reshape(-1, C, H * Wd)                               # (T, C, s)
    U, sv, Vh = torch.linalg.svd(M, full_matrices=False)
    sq = sv[:, :rank].sqrt()
    Fv = (U[:, :, :rank] * sq[:, None, :]).transpose(1, 2)      # (T, r, C)
    S = Vh[:, :rank, :] * sq[:, :, None]                         # (T, r, s)
    sgn = torch.sign(S.sum(-1, keepdim=True))
    sgn[sgn == 0] = 1
    return normalize(S * sgn, Fv * sgn)


class _Ops:
    """Linear maps of one ALS block, for all channels at once, through two GEMMs with the shared X.

    pred = Xf @ W, W[(c,s), t] = sum_r F[t,r,c] S[t,r,s];  grad wrt F[t,r,c] = sum_s S[t,r,s] G[c,s,t],
    grad wrt S[t,r,s] = sum_c F[t,r,c] G[c,s,t], with G = Xf^T R. No per-channel design matrix is formed.
    """

    def __init__(self, Xc):
        self.n, self.C, self.s = Xc.shape
        self.Xf = Xc.reshape(self.n, -1).contiguous()

    def fwd(self, S, Fv):
        return self.Xf @ torch.einsum("trc,trs->cst", Fv, S).reshape(self.C * self.s, -1)

    def adjG(self, R):
        return (self.Xf.T @ R).reshape(self.C, self.s, -1)                  # (C, s, T)

    def gradF(self, G, S):
        return torch.einsum("cst,trs->trc", G, S)

    def gradS(self, G, Fv):
        return torch.einsum("cst,trc->trs", G, Fv)


def _block_fista(ops, block, fixed, Yc, lam, w0, iters, tol=1e-6, power_iters=20):
    """FISTA for one block ("F" or "S") of all channels. Yc (n, T); lam (T,); w0 (T, r, q)."""
    n = ops.n
    if block == "F":
        A = lambda w: ops.fwd(fixed, w)                                   # noqa: E731
        At = lambda R: ops.gradF(ops.adjG(R), fixed)                     # noqa: E731
    else:
        A = lambda w: ops.fwd(w, fixed)                                   # noqa: E731
        At = lambda R: ops.gradS(ops.adjG(R), fixed)                     # noqa: E731
    v = torch.randn_like(w0)
    for _ in range(power_iters):                                          # block-diagonal over channels
        v = At(A(v))
        v = v / v.flatten(1).norm(dim=1).clamp(min=1e-30)[:, None, None]
    L = (A(v) ** 2).sum(0) / n * 1.01                                     # ||A_t v_t||^2 / n, ||v_t|| = 1
    L = L.clamp(min=1e-12)
    step = (1.0 / L)[:, None, None]
    thr = step * lam[:, None, None]
    w, z = w0.clone(), w0.clone()
    tk = torch.ones(w0.shape[0], 1, 1, device=w0.device, dtype=w0.dtype)
    for it in range(iters):
        g = At(A(z) - Yc) / n
        w_new = soft(z - step * g, thr)
        tk1 = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        z = w_new + ((tk - 1) / tk1) * (w_new - w)
        up = ((w_new - w) * g).flatten(1).sum(1)[:, None, None] > 0      # adaptive restart
        z = torch.where(up, w_new, z)
        tk1 = torch.where(up, torch.ones_like(tk1), tk1)
        done = False
        if it % 25 == 24:
            dw = (w_new - w).flatten(1).norm(dim=1) / w_new.flatten(1).norm(dim=1).clamp(min=1e-12)
            done = float(dw.max()) < tol
        w, tk = w_new, tk1
        if done:
            break
    return w


def als_path(Xtr, Ytr, S0, F0, frac_f, frac_s, n_outer=6, iters=200):
    """Warm-started ALS over a decreasing list frac_f at fixed frac_s. Xtr (n, C, s) float32 on device.

    Yields (k, S, F, b) for each frac_f[k]. Penalties: a_f,t = frac_f * max|A_t^T y_t|/n at the init design,
    likewise a_s,t with frac_s.
    """
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, (Ytr - my).contiguous()                         # Yc (n, T)
    n = Xc.shape[0]
    ops = _Ops(Xc)
    G0 = ops.adjG(Yc)
    amax_f = ops.gradF(G0, S0).abs().flatten(1).amax(1) / n
    amax_s = ops.gradS(G0, F0).abs().flatten(1).amax(1) / n
    del G0
    S, Fv = S0.clone(), F0.clone()
    for k, ff in enumerate(frac_f):
        lam_f, lam_s = ff * amax_f, frac_s * amax_s
        for _ in range(n_outer):
            Fv = _block_fista(ops, "F", S, Yc, lam_f, Fv, iters)
            S_new = _block_fista(ops, "S", Fv, Yc, lam_s, S, iters)
            # a component whose features died, or whose mask was shrunk to nothing, keeps its old mask
            keep = (Fv.abs().sum(2) == 0) | (S_new.abs().sum(2) == 0)
            S = torch.where(keep[..., None], S, S_new)
            S, Fv = normalize(S, Fv)
        b = my - torch.einsum("cs,trs,trc->t", mx, S, Fv)
        yield k, S.clone(), Fv.clone(), b
