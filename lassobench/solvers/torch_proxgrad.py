"""Approximate Lasso via "plain deep-learning optimizers" on GPU (gpu-prox session).

All (target, alpha) problems are solved simultaneously as a (B, p) weight tensor (slot-major, B = T*A,
processed in waves when memory requires). Gradients are computed manually: for a minibatch of rows b,
    G = (W X_b^T - Y_b^T) X_b / |b|.

method:
  'prox_sgd'   w <- soft(w - lr g, lr alpha)                          (full batch + lr=1/L is ISTA)
  'prox_adam'  Adam moments on the smooth part, then the prox in Adam's diagonal metric:
               w <- soft(w - lr m^/(sqrt(v^)+eps), lr alpha/(sqrt(v^)+eps))   -> exact zeros
  'subgrad'    Adam on the full objective with alpha*sign(w) subgradient; no exact zeros
  'hadamard'   w = u*v, Adam on 1/(2n)||y-Xw||^2 + alpha/2(||u||^2+||v||^2) (smooth, same minimizers)
finalize: 'prox' (one full-batch ISTA step from the final iterate: gives exact zeros, counted in time),
          'thresh' (zero |w| < thresh_rel * max|w| per slot), or 'none'.
batch_size: None = full batch; else minibatch over images, epochs = passes over the rows.
x_on_host: keep X in pinned host memory and stream minibatch rows to the GPU every step (the route when X
           does not fit in GPU memory).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from lassobench.api import PathResult, Solver
from lassobench.solvers._torch_common import (DTYPES, PhaseTimer, assemble_csc, batched_gap,
                                               default_device, device_name, extract_rows,
                                               free_gpu_bytes, lipschitz, tf32_mode, to_device)


class TorchProxGrad(Solver):
    name = "torch_proxgrad"
    device = "cuda"

    def __init__(self, tol=1e-4, max_iter=1000, method="prox_adam", lr=None, epochs=None, batch_size=None,
                 betas=(0.9, 0.999), eps=1e-8, lr_schedule="cosine", finalize="prox", thresh_rel=1e-3,
                 hadamard_init=1e-2, dtype="float32", tf32=False, x_on_host=False, mem_frac=0.8,
                 max_batch=None, seed=0, history_every=0, profile=False, device=None, **kw):
        super().__init__(tol=tol, max_iter=max_iter, **kw)
        self.method = method
        self.lr = lr
        self.epochs = epochs if epochs is not None else max_iter
        self.batch_size = batch_size
        self.betas = betas
        self.eps = eps
        self.lr_schedule = lr_schedule
        self.finalize = finalize
        self.thresh_rel = thresh_rel
        self.hadamard_init = hadamard_init
        self.dtype = dtype
        self.tf32 = tf32
        self.x_on_host = x_on_host
        self.mem_frac = mem_frac
        self.max_batch = max_batch
        self.seed = seed
        self.history_every = history_every
        self.profile = profile
        self.dev = torch.device(device) if device else default_device()

    def config(self):
        return dict(method=self.method, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                    lr_schedule=self.lr_schedule, finalize=self.finalize, dtype=self.dtype, tf32=self.tf32,
                    x_on_host=self.x_on_host)

    # ------------------------------------------------------------------
    def fit_path(self, X, Y, alphas) -> PathResult:
        t_start = time.perf_counter()
        X = np.asarray(X); Y = np.asarray(Y)
        alphas = np.asarray(alphas, np.float64)
        n, p = X.shape
        T = Y.shape[1]
        A = alphas.shape[0]
        dev = self.dev
        xdt = DTYPES[self.dtype]
        timer = PhaseTimer(dev, self.profile)
        with tf32_mode(self.tf32), torch.no_grad():
            t0 = time.perf_counter()
            if self.x_on_host:
                Xh = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
                Xh = Xh.pin_memory() if dev.type == "cuda" else Xh
                Xg = None
            else:
                Xg = to_device(X, dev, xdt)
            Yg = to_device(Y, dev, torch.float32)
            t_transfer = time.perf_counter() - t0
            t0 = time.perf_counter()
            # Lipschitz constant of the full-batch smooth part (also used by finalize='prox')
            if Xg is not None:
                L = lipschitz(Xg)
            else:
                L = self._lipschitz_streamed(Xh, dev)
            t_lip = time.perf_counter() - t0
            # slots: all (t, a) pairs
            slot_t = np.repeat(np.arange(T), A)
            slot_a = np.tile(np.arange(A), T)
            per = p * 4 * (5 if self.method in ("hadamard",) else 4) + 64 * n
            cap = self.max_batch or max(1, int(free_gpu_bytes(dev) * self.mem_frac // per))
            cap = min(cap, len(slot_t))
            pieces, gaps, hist = {}, np.full((A, T), np.nan), []
            t0 = time.perf_counter()
            for w0 in range(0, len(slot_t), cap):
                st, sa = slot_t[w0:w0 + cap], slot_a[w0:w0 + cap]
                res = self._solve_wave(Xg, Xh if Xg is None else None, Yg, alphas, st, sa, L, timer)
                for i, (r, v) in enumerate(res["rows"]):
                    pieces.setdefault(int(sa[i]), {})[int(st[i])] = (r, v)
                gaps[sa, st] = res["rel_gap"]
                hist.append(res["history"])
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_solve = time.perf_counter() - t0
        coefs = [assemble_csc(pieces.get(a, {}), p, T) for a in range(A)]
        info = dict(solver=self.name, config=self.config(), gpu=device_name(dev), rel_gap=gaps,
                    lipschitz=L, batch_capacity=int(cap), time_transfer_s=t_transfer,
                    time_lipschitz_s=t_lip, time_engine_s=t_solve, history=hist,
                    time_fit_s=time.perf_counter() - t_start)
        if self.profile:
            info["phase_s"] = dict(timer.t)
        return PathResult(coefs=coefs, info=info)

    def _lipschitz_streamed(self, Xh, dev, n_iter=100):
        n = Xh.shape[0]
        v = torch.randn(n, 1, generator=torch.Generator().manual_seed(0))
        v /= v.norm()
        lam = 0.0
        rows = max(1, int(2e8 // Xh.shape[1]))
        for _ in range(n_iter):
            u = torch.zeros(Xh.shape[1], 1, device=dev)
            for i in range(0, n, rows):
                u += Xh[i:i + rows].to(dev, non_blocking=True).T @ v[i:i + rows].to(dev)
            v2 = torch.cat([(Xh[i:i + rows].to(dev, non_blocking=True) @ u).cpu() for i in range(0, n, rows)])
            lam_new = float((v * v2).sum())
            v = v2 / v2.norm()
            if abs(lam_new - lam) < 1e-7 * lam_new:
                lam = lam_new
                break
            lam = lam_new
        return 1.01 * lam / n

    # ------------------------------------------------------------------
    def _solve_wave(self, Xg, Xh, Yg, alphas, st, sa, L, timer):
        dev = Yg.device
        n = Yg.shape[0]
        p = (Xg if Xg is not None else Xh).shape[1]
        B = len(st)
        xdt = Xg.dtype if Xg is not None else torch.float32
        tg = torch.as_tensor(st, device=dev)
        alpha = torch.as_tensor(alphas[sa, st], device=dev, dtype=torch.float32)[:, None]   # (B,1)
        Ysel = Yg.T[tg].contiguous()                                                      # (B, n)
        bs = n if self.batch_size is None else min(self.batch_size, n)
        steps_per_epoch = math.ceil(n / bs)
        n_steps = int(self.epochs) * steps_per_epoch
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        meth = self.method
        lr0 = self.lr
        if lr0 is None:
            lr0 = (1.0 / L) if meth == "prox_sgd" else 1e-3
        b1, b2 = self.betas

        W = torch.zeros(B, p, device=dev)
        if meth == "hadamard":
            U = torch.full((B, p), self.hadamard_init, device=dev)
            V = torch.zeros(B, p, device=dev)
            mU, vU = torch.zeros_like(U), torch.zeros_like(U)
            mV, vV = torch.zeros_like(V), torch.zeros_like(V)
        elif meth in ("prox_adam", "subgrad"):
            m1, m2 = torch.zeros_like(W), torch.zeros_like(W)

        def get_rows(idx):
            if Xg is not None:
                return Xg if idx is None else Xg.index_select(0, idx)
            if idx is None:
                return Xh.to(dev, non_blocking=True)
            return Xh[idx.cpu()].to(dev, non_blocking=True)

        def grad_smooth(Wc, idx):
            Xb = get_rows(idx)
            yb = Ysel if idx is None else Ysel.index_select(1, idx)
            R = (Wc.to(xdt) @ Xb.T).float() - yb               # (B, bs)
            return (R.to(xdt) @ Xb).float() / Xb.shape[0]      # (B, p)

        history = []
        step = 0
        for ep in range(int(self.epochs)):
            perm = None if bs == n else torch.randperm(n, generator=gen).to(dev)
            for k in range(steps_per_epoch):
                idx = None if perm is None else perm[k * bs:(k + 1) * bs]
                if self.lr_schedule == "cosine":
                    lr = lr0 * 0.5 * (1 + math.cos(math.pi * step / max(1, n_steps)))
                elif self.lr_schedule == "invsqrt" and bs < n:
                    lr = lr0 / math.sqrt(1 + step / steps_per_epoch)
                else:
                    lr = lr0
                step += 1
                with timer("grad"):
                    if meth == "hadamard":
                        Wc = U * V
                    else:
                        Wc = W
                    G = grad_smooth(Wc, idx)
                with timer("update"):
                    if meth == "prox_sgd":
                        W.add_(G, alpha=-lr)
                        th = lr * alpha
                        W.sub_(torch.clamp(W, -th, th))
                    elif meth == "prox_adam":
                        m1.mul_(b1).add_(G, alpha=1 - b1)
                        m2.mul_(b2).addcmul_(G, G, value=1 - b2)
                        c1, c2 = 1 - b1 ** step, 1 - b2 ** step
                        D = (m2 / c2).sqrt_().add_(self.eps)       # diagonal metric
                        W.addcdiv_(m1, D, value=-lr / c1)
                        th = (lr * alpha) / D
                        W.sub_(torch.minimum(torch.maximum(W, -th), th))
                        del D, th
                    elif meth == "subgrad":
                        G.add_(alpha * torch.sign(W))
                        m1.mul_(b1).add_(G, alpha=1 - b1)
                        m2.mul_(b2).addcmul_(G, G, value=1 - b2)
                        c1, c2 = 1 - b1 ** step, 1 - b2 ** step
                        W.addcdiv_(m1, (m2 / c2).sqrt_().add_(self.eps), value=-lr / c1)
                    elif meth == "hadamard":
                        gU = G * V + alpha * U
                        gV = G * U + alpha * V
                        for P_, g_, m_, v_ in ((U, gU, mU, vU), (V, gV, mV, vV)):
                            m_.mul_(b1).add_(g_, alpha=1 - b1)
                            v_.mul_(b2).addcmul_(g_, g_, value=1 - b2)
                            c1, c2 = 1 - b1 ** step, 1 - b2 ** step
                            P_.addcdiv_(m_, (v_ / c2).sqrt_().add_(self.eps), value=-lr / c1)
                        del gU, gV
                    else:
                        raise ValueError(meth)
            if self.history_every and ((ep + 1) % self.history_every == 0):
                Wc = U * V if meth == "hadamard" else W
                g, rel, nnz = self._gap(Xg, Xh, Wc, Ysel, alpha, n)
                history.append(dict(epoch=ep + 1, med_rel_gap=float(rel.median()), max_rel_gap=float(rel.max()),
                                    med_nnz=float(nnz.float().median())))
        if meth == "hadamard":
            W = U * V
            del U, V, mU, vU, mV, vV
        with timer("finalize"):
            if self.finalize == "prox":
                G = grad_smooth(W, None)
                W.add_(G, alpha=-1.0 / L)
                th = alpha / L
                W.sub_(torch.clamp(W, -th, th))
            elif self.finalize == "thresh":
                mx = W.abs().amax(1, keepdim=True)
                W.masked_fill_(W.abs() < self.thresh_rel * mx, 0.0)
        with timer("gap"):
            _, rel, _ = self._gap(Xg, Xh, W, Ysel, alpha, n)
        with timer("extract"):
            rows = extract_rows(W, torch.arange(B, device=dev))
        return dict(rows=rows, rel_gap=rel.cpu().numpy(), history=history)

    def _gap(self, Xg, Xh, W, Ysel, alpha, n):
        dev = W.device
        Xf = Xg if Xg is not None else Xh.to(dev)
        xdt = Xf.dtype
        R = Ysel - (W.to(xdt) @ Xf.T).float()
        C = (R.to(xdt) @ Xf).float()
        Cinf = torch.linalg.vector_norm(C, ord=float("inf"), dim=1).double()
        del C
        l1 = torch.linalg.vector_norm(W, ord=1, dim=1).double()
        ninv = torch.full((W.shape[0],), 1.0 / n, device=dev, dtype=torch.float64)
        ynorm2 = (Ysel.double() ** 2).sum(1)
        gap, rel = batched_gap(R, Cinf, l1, Ysel, alpha[:, 0].double(), ninv, ynorm2)
        return gap, rel, (W != 0).sum(1)


class _Preset(TorchProxGrad):
    preset: dict = {}

    def __init__(self, **kw):
        kw = {**self.preset, **kw}
        super().__init__(**kw)


def _make(name, **preset):
    return type(name, (_Preset,), dict(name=name, preset=preset))


SOLVERS = {
    "torch_proxgrad": TorchProxGrad,                                     # prox-Adam, full batch
    "torch_prox_adam": _make("torch_prox_adam", method="prox_adam"),
    "torch_prox_adam_mb": _make("torch_prox_adam_mb", method="prox_adam", batch_size=128),
    "torch_prox_sgd": _make("torch_prox_sgd", method="prox_sgd"),       # == ISTA at lr=1/L
    "torch_subgrad_adam": _make("torch_subgrad_adam", method="subgrad"),
    "torch_hadamard_adam": _make("torch_hadamard_adam", method="hadamard"),
}
