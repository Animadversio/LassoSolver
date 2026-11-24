"""torch_admm: batched ADMM for the Lasso path with the matrix-inversion lemma.

    min 1/(2n)||y - X w||^2 + alpha ||z||_1   s.t. w = z          (scaled dual u)
    w <- (X^T X/n + rho I)^{-1} (X^T y/n + rho (z - u))
       = (v - X^T (X X^T + n rho I)^{-1} X v) / rho,   v = X^T y/n + rho (z - u)
    z <- S_{alpha/rho}(gamma w + (1-gamma) z + u)       (over-relaxation gamma)
    u <- u + gamma w + (1-gamma) z_old - z

The n x n inverse is applied through ONE eigendecomposition X X^T = Q diag(l) Q^T shared by all
targets, alphas and rho values: (X X^T + n rho_b I)^{-1} = Q diag(1/(l + n rho_b)) Q^T. So adaptive
rho is per problem and free (no refactorization), unlike the Cholesky variant whose every rho change
costs an n^3/3 refactorization (shared rho only). Each iteration costs two O(n p B) matmuls
(X @ (z - u) and X^T @ (.)) plus two O(n^2 B); exact zeros come from z.

Stopping: every `check_every` iterations the duality gap of z (residual rescaling, X^T R costs one
more matmul pair) is evaluated; problems at rel gap <= tol are frozen and dropped from the batch.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from lassobench.api import PathResult, Solver
from lassobench.solvers._ws_common import Timer, gap_from_stats, to_csc_list


def _admm_path(X, Y, alphas, tol, max_iter=2000, rho0=None, gamma=1.6, adaptive=True, mu=10.0, tau=2.0,
               check_every=10, factor="eigh", device="cuda", verbose=False, dtype="float32"):
    tm = Timer()
    t_start = time.perf_counter()
    dev = torch.device(device)
    dt = getattr(torch, dtype)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    with tm("h2d"):
        Xd = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev).to(dt)
        Yd = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(dev).to(dt)
    n, p = Xd.shape
    T = Yd.shape[1]
    A = alphas.shape[0]
    al = torch.as_tensor(np.asarray(alphas, dtype=np.float64), device=dev)
    with tm("prepare"):
        K64 = (Xd @ Xd.T).double()
        if factor == "eigh":
            lam_e, Q = torch.linalg.eigh(K64)
            lam_e = torch.clamp(lam_e, min=0)
            Qd = Q.to(dt)
        else:
            raise ValueError(factor)
        Xty = (Xd.T @ Yd) / n                       # (p, T)
        KY = (K64 @ Yd.double()).to(dt) / n          # X X^T y / n  (n, T)
        yy = (Yd.double() ** 2).sum(0)
    if rho0 is None:
        # scale-aware default: mean eigenvalue of X^T X / n restricted to its range ~ trace/n / n
        rho_init = float(lam_e.sum() / n / n)
    else:
        rho_init = float(rho0)
    z = torch.zeros(p, T, dtype=dt, device=dev)
    u = torch.zeros(p, T, dtype=dt, device=dev)
    rho = torch.full((T,), rho_init, dtype=torch.float64, device=dev)
    rows, cols, vals = [], [], []
    info = dict(iters=[], per_alpha=[])

    def wstep(zb, ub, rb, idx):
        """w = (v - X^T Q D Q^T X v)/rho with v = X^T y/n + rho (z - u)."""
        r32 = rb.to(dt)
        Xv = KY[:, idx] + r32[None, :] * (Xd @ (zb - ub))           # X v   (n, b)
        D = 1.0 / (lam_e[:, None] + n * rb[None, :])                 # (n, b)
        s = Qd @ ((Qd.T @ Xv) * D.to(dt))
        return (Xty[:, idx] + r32[None, :] * (zb - ub) - Xd.T @ s) / r32[None, :]

    def gap_of(zb, idx, lam):
        R = Yd[:, idx] - Xd @ zb
        qmax = (Xd.T @ R).abs().max(0).values.double()
        R64 = R.double()
        rr = (R64 * R64).sum(0)
        ry = (R64 * Yd[:, idx].double()).sum(0)
        l1 = zb.double().abs().sum(0)
        return gap_from_stats(rr, ry, l1, qmax, lam, yy[idx])

    for a in range(A):
        lam = al[a] * n                                             # unscaled lam (T,)
        alpha = al[a].to(dt)
        act = torch.arange(T, device=dev)
        with tm("check"):
            g = gap_of(z, act, lam)
        act = act[g > tol]
        it = 0
        while act.numel() > 0 and it < max_iter:
            zb, ub, rb = z[:, act], u[:, act], rho[act]
            ab = alpha[act]
            with tm("iterate"):
                for _ in range(check_every):
                    w = wstep(zb, ub, rb, act)
                    wh = gamma * w + (1 - gamma) * zb
                    v = wh + ub
                    th = (ab / rb.to(dt))[None, :]
                    znew = torch.sign(v) * torch.clamp(v.abs() - th, min=0)
                    ub = ub + wh - znew
                    if adaptive:
                        rp = torch.linalg.vector_norm((w - znew).double(), dim=0)
                        rd = rb * torch.linalg.vector_norm((znew - zb).double(), dim=0)
                        up = rp > mu * rd
                        dn = rd > mu * rp
                        fac = torch.where(up, torch.full_like(rb, tau), torch.where(dn, torch.full_like(rb, 1 / tau), torch.ones_like(rb)))
                        ub = ub / fac.to(dt)[None, :]
                        rb = rb * fac
                    zb = znew
                    it += 1
            z[:, act], u[:, act], rho[act] = zb, ub, rb
            with tm("check"):
                g = gap_of(zb, act, lam[act])
            act = act[g > tol]
        with tm("extract"):
            nz = z != 0
            r_, c_ = nz.nonzero(as_tuple=True)
            rows.append(r_)
            cols.append(c_)
            vals.append(z[r_, c_].double())
        info["iters"].append(it)
        info["per_alpha"].append(dict(iters=it, n_unconv=int(act.numel()), nnz_max=int(nz.sum(0).max())))
        if verbose:
            print(f"a={a} iters={it} unconv={act.numel()} nnz_max={int(nz.sum(0).max())} rho_med={float(rho.median()):.3g}", flush=True)
    with tm("d2h"):
        rows = [r.cpu().numpy() for r in rows]
        cols = [c.cpu().numpy() for c in cols]
        vals = [v.cpu().numpy() for v in vals]
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    info.update(timings=tm.t, time_wall=time.perf_counter() - t_start, gpu=torch.cuda.get_device_name(dev),
                peak_gpu_mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9, total_iters=int(sum(info["iters"])))
    return rows, cols, vals, info


class TorchADMM(Solver):
    name = "torch_admm"
    device = "cuda"

    def fit_path(self, X, Y, alphas):
        n, p = X.shape
        T = Y.shape[1]
        rows, cols, vals, info = _admm_path(X, Y, np.asarray(alphas), self.tol, max_iter=self.max_iter, **self.kw)
        return PathResult(coefs=to_csc_list(rows, cols, vals, len(rows), p, T), info=info)


SOLVERS = {"torch_admm": TorchADMM}
