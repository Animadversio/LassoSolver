"""Shared torch helpers for the GPU Lasso solvers (owned by the gpu-prox session).

Other sessions may import from here but should not edit it.

Conventions: X is (n, p) on the device, Y is (n, T). Weights are stored feature-major as (p, B)
where B indexes a batch of independent problems ("slots"), each with its own target column,
alpha and (optionally) training-row mask. Objective per slot b:

    1/(2 n_b) ||m_b * (y_b - X w_b)||^2 + alpha_b ||w_b||_1

where m_b is a 0/1 row mask (all ones for a plain fit) and n_b = sum(m_b).
"""
from __future__ import annotations

import contextlib
import time

import numpy as np
import scipy.sparse as sp
import torch

DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16,
          "float16": torch.float16}


def device_name(dev: torch.device | str | None = None) -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(dev if dev is not None else 0)
    return "cpu"


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def tf32_mode(enabled: bool):
    """Temporarily set TF32 for matmuls (affects float32 matmuls on Ampere+)."""
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn


def sync(dev):
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.synchronize(dev)


class PhaseTimer:
    """Accumulates wall time per phase. With `enabled=False` it is a no-op (no syncs)."""

    def __init__(self, dev, enabled: bool = False):
        self.dev, self.enabled = dev, enabled
        self.t: dict[str, float] = {}
        self._name, self._t0 = None, 0.0

    @contextlib.contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        sync(self.dev)
        t0 = time.perf_counter()
        yield
        sync(self.dev)
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0


def to_device(A: np.ndarray, dev, dtype=torch.float32, chunk_rows: int | None = None) -> torch.Tensor:
    """Host -> device copy in row chunks (bounds host temporaries when casting large arrays)."""
    A = np.asarray(A)
    out = torch.empty(A.shape, dtype=dtype, device=dev)
    if chunk_rows is None:
        chunk_rows = max(1, int(2e8 // max(1, A.shape[1] if A.ndim > 1 else 1)))
    for i in range(0, A.shape[0], chunk_rows):
        blk = torch.from_numpy(np.ascontiguousarray(A[i:i + chunk_rows]))
        out[i:i + chunk_rows].copy_(blk.to(dtype) if dtype != torch.bfloat16 else blk.float().to(dtype),
                                    non_blocking=False)
    return out


@torch.no_grad()
def lipschitz(X: torch.Tensor, mask: torch.Tensor | None = None, n_iter: int = 300, tol: float = 1e-7,
              safety: float = 1.01, seed: int = 0) -> float:
    """Top eigenvalue of X_m^T X_m / n_m (= of X_m X_m^T / n_m), by power iteration in n-space.

    mask: optional (n,) 0/1 float tensor selecting training rows (no copy of X).
    Returns safety * lambda_max (power iteration approaches from below).
    """
    n = X.shape[0]
    g = torch.Generator(device=X.device).manual_seed(seed)
    v = torch.randn(n, 1, device=X.device, dtype=torch.float32, generator=g)
    m = mask.reshape(n, 1).to(torch.float32) if mask is not None else None
    n_eff = float(m.sum()) if m is not None else float(n)
    if m is not None:
        v = v * m
    v = v / v.norm()
    lam = 0.0
    xd = X.dtype
    for _ in range(n_iter):
        u = X.T @ v.to(xd)                    # (p, 1)
        v2 = (X @ u).float()                  # (n, 1)
        if m is not None:
            v2 = v2 * m
        lam_new = float((v * v2).sum())
        v = v2 / v2.norm()
        if abs(lam_new - lam) <= tol * abs(lam_new):
            lam = lam_new
            break
        lam = lam_new
    return safety * lam / n_eff


@torch.no_grad()
def batched_gap(R: torch.Tensor, C_inf: torch.Tensor, l1: torch.Tensor, Ysel: torch.Tensor,
                alpha: torch.Tensor, ninv: torch.Tensor, ynorm2: torch.Tensor):
    """Duality gap per slot, dual point by residual rescaling (same as api.duality_gap).

    R: (B, n) masked residual y - Xw (slot-major). C_inf: (B,) ||X^T R||_inf. l1: (B,) ||w||_1.
    All (B,) inputs are float64. Returns (gap, rel_gap) float64, rel_gap = gap / (||y||^2/(2n)).
    """
    Rd = R.double()
    rr = (Rd * Rd).sum(1)
    yr = (Ysel.double() * Rd).sum(1)
    corr = C_inf * ninv
    scale = torch.clamp(alpha / torch.clamp(corr, min=1e-300), max=1.0)
    primal = 0.5 * rr * ninv + alpha * l1
    # ||y||^2 - ||y - s r||^2 = 2 s <y,r> - s^2 ||r||^2
    dual = 0.5 * ninv * (2 * scale * yr - scale * scale * rr)
    gap = primal - dual
    rel = gap / torch.clamp(0.5 * ynorm2 * ninv, min=1e-300)
    return gap, rel


@torch.no_grad()
def extract_rows(W: torch.Tensor, rows: torch.Tensor):
    """Pull sparse slot rows W[rows, :] of a slot-major (B, p) tensor to host.

    Returns list of (feature_idx int32, values float32), one per requested row."""
    if rows.numel() == 0:
        return []
    p = W.shape[1]
    step = max(1, int((2**31 - 1) // max(1, p)) // 2)   # torch.nonzero is limited to INT_MAX elements
    if rows.numel() > step:
        out = []
        for i in range(0, rows.numel(), step):
            out += extract_rows(W, rows[i:i + step])
        return out
    sub = W.index_select(0, rows)                     # (k, p)
    nz = sub.nonzero()                                # row-major: sorted by column then feature
    vals = sub[nz[:, 0], nz[:, 1]].float().cpu().numpy()
    nz = nz.cpu().numpy()
    k = sub.shape[0]
    counts = np.bincount(nz[:, 0], minlength=k)
    splits = np.cumsum(counts)[:-1]
    rows = np.split(nz[:, 1].astype(np.int32), splits)
    vs = np.split(vals, splits)
    return list(zip(rows, vs))


def assemble_csc(pieces: dict, p: int, T: int) -> sp.csc_matrix:
    """pieces: {t: (row_idx, values)} -> CSC (p, T). Missing targets are all-zero columns."""
    indptr = np.zeros(T + 1, dtype=np.int64)
    idx_list, val_list = [], []
    for t in range(T):
        if t in pieces:
            r, v = pieces[t]
            o = np.argsort(r, kind="stable")
            idx_list.append(r[o]); val_list.append(v[o])
            indptr[t + 1] = len(r)
        else:
            idx_list.append(np.zeros(0, np.int32)); val_list.append(np.zeros(0, np.float32))
    indptr = np.cumsum(indptr)
    return sp.csc_matrix((np.concatenate(val_list).astype(np.float32),
                          np.concatenate(idx_list).astype(np.int32), indptr), shape=(p, T))


def free_gpu_bytes(dev) -> int:
    if isinstance(dev, torch.device) and dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        free, _ = torch.cuda.mem_get_info(idx)
        return int(free + torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx))
    return int(64e9)


# ----------------------------------------------------------------------------- fused prox (Triton)
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _prox_restart_kernel(G, WP, W, BETA, STEP, THR, S, p, BLOCK: tl.constexpr, RESTART: tl.constexpr):
        # z = (1+beta) w - beta w_prev ; G <- soft(z - step*G, thr) ; S[b] += <z - wn, wn - w>
        b = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        msk = offs < p
        base = b.to(tl.int64) * p
        g = tl.load(G + base + offs, mask=msk, other=0.0)
        w = tl.load(W + base + offs, mask=msk, other=0.0)
        wp = tl.load(WP + base + offs, mask=msk, other=0.0)
        beta = tl.load(BETA + b)
        step = tl.load(STEP + b)
        thr = tl.load(THR + b)
        z = w + beta * (w - wp)
        u = z - step * g
        wn = tl.where(u > thr, u - thr, tl.where(u < -thr, u + thr, 0.0))
        tl.store(G + base + offs, wn, mask=msk)
        if RESTART:
            s = tl.sum((z - wn) * (wn - w), axis=0)
            tl.atomic_add(S + b, s)

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


def fused_prox_available(t: torch.Tensor) -> bool:
    return _HAVE_TRITON and t.is_cuda and t.dtype in (torch.float32,)


def fused_prox_(G, Wp, W, beta, step, thr, s_out, restart=True, block=2048):
    """One fused FISTA prox pass, in place on G (which holds the gradient at z on entry):

        z = (1+beta) W - beta Wp;  G <- soft(z - step*G, thr);  s_out[b] = <z - G, G - W>  (restart test)

    G, Wp, W: contiguous (B, p) float32. beta, step, thr: (B,) float32. s_out: (B,) float32 (overwritten).
    z is never materialized (the gradient only needs Xz, which lives in n-space)."""
    B, p = G.shape
    s_out.zero_()
    grid = (B, triton.cdiv(p, block))
    _prox_restart_kernel[grid](G, Wp, W, beta, step, thr, s_out, p, BLOCK=block, RESTART=restart)
    return s_out
