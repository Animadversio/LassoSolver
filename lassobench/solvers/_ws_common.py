"""Shared GPU pieces for the working-set (torch_ws) and ADMM (torch_admm) Lasso solvers.

Conventions (all problems share one X, problem b = (target, fold)):
    min_w 1/(2 n_b) ||y_b - X_b w||^2 + alpha_b ||w||_1
is solved in the unscaled form  0.5 ||y_b - X_b w||^2 + lam_b ||w||_1  with lam_b = n_b * alpha_b.
X is kept on the GPU as given, (n, p) row-major; the full correlation is computed as R^T X in
column chunks (cuBLAS handles the strided slices), so no transposed copy is ever made.

Contents
- `cd_gram_batched`: a CUDA kernel (cupy RawKernel / NVRTC) running Gram-based cyclic coordinate
  descent with one thread block per problem. Each block reads its working set's Gram entries from a
  shared union Gram (positions `pos`), keeps w and q = c - G w in shared memory, alternates full and
  active-only epochs, and stops on the restricted duality gap.
- `full_corr_stats`: the expensive step, X^T R for all problems at once, chunked over features;
  returns max |x_j^T r_b| (dual scaling), the top-M features per problem (working-set candidates)
  and the strong-rule counts, without ever materializing (p, B).
- `gap_from_stats`: duality gap with the residual-rescaling dual point.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

_KERNEL_SRC = r"""
extern "C" __global__ void cd_gram(
    const GT* __restrict__ G, const long long* __restrict__ goff, const long long ldg,
    const int* __restrict__ pos, const int* __restrict__ kk, const int K,
    const double* __restrict__ c, const double* __restrict__ dg,
    double* __restrict__ w_io,
    const double* __restrict__ lam, const double* __restrict__ yy, const double* __restrict__ tol,
    const int max_ep, const int act_passes,
    int* __restrict__ ep_out, double* __restrict__ gap_out,
    const int anderson, double* __restrict__ snaps, double* __restrict__ wacc, double* __restrict__ qacc)
{
    extern __shared__ double sh[];
    const int b = blockIdx.x;
    const int tid = threadIdx.x, bs = blockDim.x;
    const int k = kk[b];
    double* w = sh;                  // K
    double* q = sh + K;              // K
    double* d = sh + 2 * K;          // K  (Gram diagonal)
    int* ps = (int*)(sh + 3 * K);    // K  (positions in union Gram)
    int* act = ps + K;               // K  (active list)
    __shared__ double red[10][32];
    __shared__ double s_coef[4];
    __shared__ int s_ok;
    __shared__ double s_delta;
    __shared__ int s_nact, s_stop;

    const double* cb = c + (long long)b * K;
    const GT* Gb = G + goff[b];
    for (int i = tid; i < k; i += bs) {
        w[i] = w_io[(long long)b * K + i];
        d[i] = dg[(long long)b * K + i];
        ps[i] = pos[(long long)b * K + i];
        q[i] = cb[i];
    }
    __syncthreads();
    if (tid == 0) {
        int na = 0;
        for (int j = 0; j < k; ++j) if (w[j] != 0.0) act[na++] = j;
        s_nact = na; s_stop = 0;
    }
    __syncthreads();
    // q = c - G w   (only over nonzero w)
    for (int a = 0; a < s_nact; ++a) {
        const int j = act[a];
        const GT* row = Gb + (long long)ps[j] * ldg;
        const double wj = w[j];
        for (int i = tid; i < k; i += bs) q[i] -= (double)row[ps[i]] * wj;
    }
    __syncthreads();
    const double lb = lam[b], yb = yy[b], tb = tol[b];
    int ep = 0, nsnap = 0;
    double gap_rel = 1e300;
    while (true) {
        // ---- restricted duality gap (residual rescaling) ----
        double cw = 0, qw = 0, l1 = 0, qm = 0;
        for (int i = tid; i < k; i += bs) {
            cw += cb[i] * w[i]; qw += q[i] * w[i]; l1 += fabs(w[i]); qm = fmax(qm, fabs(q[i]));
        }
        for (int o = 16; o > 0; o >>= 1) {
            cw += __shfl_down_sync(0xffffffff, cw, o); qw += __shfl_down_sync(0xffffffff, qw, o);
            l1 += __shfl_down_sync(0xffffffff, l1, o); qm = fmax(qm, __shfl_down_sync(0xffffffff, qm, o));
        }
        if ((tid & 31) == 0) { red[0][tid >> 5] = cw; red[1][tid >> 5] = qw; red[2][tid >> 5] = l1; red[3][tid >> 5] = qm; }
        __syncthreads();
        if (tid == 0) {
            const int nw = (bs + 31) >> 5;
            cw = 0; qw = 0; l1 = 0; qm = 0;
            for (int i = 0; i < nw; ++i) { cw += red[0][i]; qw += red[1][i]; l1 += red[2][i]; qm = fmax(qm, red[3][i]); }
            const double rr = yb - cw - qw, ry = yb - cw;
            const double s = (qm > lb) ? lb / qm : 1.0;
            const double P = 0.5 * rr + lb * l1;
            const double D = s * ry - 0.5 * s * s * rr;
            gap_rel = 2.0 * (P - D) / fmax(yb, 1e-300);
            s_stop = (gap_rel <= tb || ep >= max_ep) ? 1 : 0;
            red[0][0] = gap_rel;
        }
        __syncthreads();
        if (s_stop) { gap_rel = red[0][0]; break; }
        // ---- one full epoch, then `act_passes` epochs over the active set ----
        for (int pass = 0; pass <= act_passes; ++pass) {
            int nj;
            if (pass == 0) nj = k;
            else {
                if (tid == 0) { int na = 0; for (int j = 0; j < k; ++j) if (w[j] != 0.0) act[na++] = j; s_nact = na; }
                __syncthreads();
                nj = s_nact;
                if (nj == 0) break;
            }
            for (int jj = 0; jj < nj; ++jj) {
                const int j = (pass == 0) ? jj : act[jj];
                if (tid == 0) {
                    const double wj = w[j], dj = d[j];
                    const double z = wj + q[j] / dj, th = lb / dj;
                    const double wn = (z > th) ? z - th : ((z < -th) ? z + th : 0.0);
                    s_delta = wn - wj;
                    w[j] = wn;
                }
                __syncthreads();
                const double dl = s_delta;
                if (dl != 0.0) {
                    const GT* row = Gb + (long long)ps[j] * ldg;
                    for (int i = tid; i < k; i += bs) q[i] -= (double)row[ps[i]] * dl;
                }
                __syncthreads();
            }
            ++ep;
        }
        // ---- Anderson extrapolation over the last 5 group iterates (celer-style) ----
        if (anderson) {
            double* sb = snaps + (long long)b * 5 * K;
            for (int i = tid; i < k; i += bs) sb[nsnap * K + i] = w[i];
            ++nsnap;
            if (nsnap == 5) {
                nsnap = 0;
                double acc[10];
                for (int m = 0; m < 10; ++m) acc[m] = 0;
                for (int i = tid; i < k; i += bs) {
                    double u[4];
                    for (int m = 0; m < 4; ++m) u[m] = sb[(m + 1) * K + i] - sb[m * K + i];
                    int t = 0;
                    for (int m = 0; m < 4; ++m) for (int l = m; l < 4; ++l) acc[t++] += u[m] * u[l];
                }
                for (int m = 0; m < 10; ++m) {
                    double v = acc[m];
                    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
                    if ((tid & 31) == 0) red[m][tid >> 5] = v;
                }
                __syncthreads();
                if (tid == 0) {
                    const int nw = (bs + 31) >> 5;
                    double M[4][5];
                    int t = 0;
                    double tr = 0;
                    for (int m = 0; m < 4; ++m) for (int l = m; l < 4; ++l) {
                        double v = 0; for (int i = 0; i < nw; ++i) v += red[t][i];
                        M[m][l] = v; M[l][m] = v; ++t;
                    }
                    for (int m = 0; m < 4; ++m) tr += M[m][m];
                    for (int m = 0; m < 4; ++m) { M[m][m] += 1e-10 * tr + 1e-300; M[m][4] = 1.0; }
                    int ok = tr > 0;
                    for (int m = 0; m < 4 && ok; ++m) {         // Gaussian elimination, partial pivot
                        int piv = m;
                        for (int l = m + 1; l < 4; ++l) if (fabs(M[l][m]) > fabs(M[piv][m])) piv = l;
                        if (fabs(M[piv][m]) < 1e-300) { ok = 0; break; }
                        for (int l = 0; l < 5; ++l) { double tmp = M[m][l]; M[m][l] = M[piv][l]; M[piv][l] = tmp; }
                        for (int l = m + 1; l < 4; ++l) {
                            const double f = M[l][m] / M[m][m];
                            for (int h = m; h < 5; ++h) M[l][h] -= f * M[m][h];
                        }
                    }
                    double z[4], sz = 0;
                    for (int m = 3; m >= 0 && ok; --m) {
                        double v = M[m][4];
                        for (int l = m + 1; l < 4; ++l) v -= M[m][l] * z[l];
                        z[m] = v / M[m][m]; sz += z[m];
                    }
                    if (ok && isfinite(sz) && fabs(sz) > 1e-300) { for (int m = 0; m < 4; ++m) s_coef[m] = z[m] / sz; }
                    else ok = 0;
                    s_ok = ok;
                }
                __syncthreads();
                if (s_ok) {
                    double* wa = wacc + (long long)b * K;
                    double* qa = qacc + (long long)b * K;
                    for (int i = tid; i < k; i += bs) {
                        double v = 0;
                        for (int m = 0; m < 4; ++m) v += s_coef[m] * sb[(m + 1) * K + i];
                        wa[i] = v; qa[i] = cb[i];
                    }
                    __syncthreads();
                    for (int j = 0; j < k; ++j) {
                        const double wj = wa[j];
                        if (wj != 0.0) {
                            const GT* row = Gb + (long long)ps[j] * ldg;
                            for (int i = tid; i < k; i += bs) qa[i] -= (double)row[ps[i]] * wj;
                        }
                    }
                    __syncthreads();
                    double v[6] = {0, 0, 0, 0, 0, 0};   // c.wa, qa.wa, |wa|, c.w, q.w, |w|
                    for (int i = tid; i < k; i += bs) {
                        v[0] += cb[i] * wa[i]; v[1] += qa[i] * wa[i]; v[2] += fabs(wa[i]);
                        v[3] += cb[i] * w[i]; v[4] += q[i] * w[i]; v[5] += fabs(w[i]);
                    }
                    for (int m = 0; m < 6; ++m) {
                        double x = v[m];
                        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
                        if ((tid & 31) == 0) red[m][tid >> 5] = x;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        const int nw = (bs + 31) >> 5;
                        double r[6] = {0, 0, 0, 0, 0, 0};
                        for (int m = 0; m < 6; ++m) for (int i = 0; i < nw; ++i) r[m] += red[m][i];
                        const double Pa = 0.5 * (yb - r[0] - r[1]) + lb * r[2];
                        const double Pc = 0.5 * (yb - r[3] - r[4]) + lb * r[5];
                        s_ok = Pa < Pc;
                    }
                    __syncthreads();
                    if (s_ok) for (int i = tid; i < k; i += bs) { w[i] = wa[i]; q[i] = qa[i]; }
                    __syncthreads();
                }
            }
        }
    }
    for (int i = tid; i < k; i += bs) w_io[(long long)b * K + i] = w[i];
    if (tid == 0) { ep_out[b] = ep; gap_out[b] = gap_rel; }
}
"""

_kernel_cache = {}


def _get_kernel(gt="double"):
    import cupy as cp
    key = (cp.cuda.Device().id, gt)
    if key not in _kernel_cache:
        k = cp.RawKernel("typedef %s GT;\n" % gt + _KERNEL_SRC, "cd_gram", options=("--std=c++14",))
        k.max_dynamic_shared_size_bytes = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"] - 8192
        _kernel_cache[key] = k
    return _kernel_cache[key]


def smem_per_k() -> int:
    return 3 * 8 + 2 * 4


def max_k_smem() -> int:
    """Largest working-set size the CD kernel can hold in shared memory on this device."""
    import cupy as cp
    lim = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
    return (lim - 8192) // smem_per_k()


def cd_gram_batched(G, goff, ldg, pos, kk, c, dg, w, lam, yy, tol, max_ep=2000, act_passes=4, anderson=True):
    """Run the batched Gram CD kernel in place on `w` (B, K) float64.

    G: float32 or float64 tensor holding all union Grams (flat). goff: (B,) int64 element offset of each
    problem's Gram. ldg: row stride. pos: (B, K) int32 positions into that Gram. kk: (B,) int32
    working-set sizes (entries beyond kk[b] are ignored). c: (B, K) float64 x_j^T y_b. dg: (B, K)
    float64 Gram diagonal. lam, yy, tol: (B,) float64. Returns (epochs (B,), restricted rel gap (B,)).
    """
    import cupy as cp
    B, K = w.shape
    if B == 0:
        return torch.zeros(0, dtype=torch.int32, device=w.device), torch.zeros(0, dtype=torch.float64, device=w.device)
    kern = _get_kernel("double" if G.dtype == torch.float64 else "float")
    smem = K * smem_per_k()
    ep = torch.empty(B, dtype=torch.int32, device=w.device)
    gap = torch.empty(B, dtype=torch.float64, device=w.device)
    bs = 128 if K <= 512 else 256
    if anderson:
        snaps = torch.empty(B * 5 * K, dtype=torch.float64, device=w.device)
        wacc = torch.empty(B * K, dtype=torch.float64, device=w.device)
        qacc = torch.empty(B * K, dtype=torch.float64, device=w.device)
    else:
        snaps = wacc = qacc = torch.empty(1, dtype=torch.float64, device=w.device)
    a = lambda t: cp.asarray(t.contiguous()) if not t.is_contiguous() else cp.asarray(t)
    torch.cuda.current_stream().synchronize()
    kern((B,), (bs,), (a(G), a(goff), np.int64(ldg), a(pos), a(kk), np.int32(K), a(c), a(dg), a(w),
                       a(lam), a(yy), a(tol), np.int32(max_ep), np.int32(act_passes), a(ep), a(gap),
                       np.int32(1 if anderson else 0), a(snaps), a(wacc), a(qacc)),
         shared_mem=smem)
    cp.cuda.get_current_stream().synchronize()
    return ep, gap


_KERNEL2_SRC = r"""
// v2: active passes on a compacted fp32 Gram of the active set (coalesced rows, register prefetch);
// exact q refresh from the (fp64) union Gram before every gap check; Anderson within the active phase.
__device__ __forceinline__ void block_reduce(double* v, int nv, double (*red)[32], int tid, int bs, int* is_max) {
    for (int m = 0; m < nv; ++m) {
        double x = v[m];
        for (int o = 16; o > 0; o >>= 1) {
            double y = __shfl_down_sync(0xffffffff, x, o);
            x = is_max[m] ? fmax(x, y) : x + y;
        }
        if ((tid & 31) == 0) red[m][tid >> 5] = x;
    }
    __syncthreads();
    if (tid == 0) {
        const int nw = (bs + 31) >> 5;
        for (int m = 0; m < nv; ++m) {
            double x = red[m][0];
            for (int i = 1; i < nw; ++i) x = is_max[m] ? fmax(x, red[m][i]) : x + red[m][i];
            red[m][0] = x;
        }
    }
    __syncthreads();
}

template <int MP>
__global__ void cd2(
    const GT* __restrict__ G, const long long* __restrict__ goff, const long long ldg,
    const int* __restrict__ pos, const int* __restrict__ kk, const int K,
    const double* __restrict__ c, const double* __restrict__ dg, double* __restrict__ w_io,
    const double* __restrict__ lam, const double* __restrict__ yy, const double* __restrict__ tol,
    const int max_ep, const int act_passes, int* __restrict__ ep_out, double* __restrict__ gap_out,
    const int anderson, double* __restrict__ snaps, float* __restrict__ GA, const int acap, const int snap_every,
    const double* __restrict__ l2v)
{
    extern __shared__ double sh[];
    const int b = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const int k = kk[b];
    double* w = sh; double* q = sh + K; double* di = sh + 2 * K;
    int* ps = (int*)(sh + 3 * K); int* act = ps + K;
    __shared__ double red[10][32];
    __shared__ double s_delta, s_coef[4];
    __shared__ int s_na, s_stop, s_ok;
    const double* cb = c + (long long)b * K;
    const GT* Gb = G + goff[b];
    float* GAb = GA + (long long)b * acap * acap;
    double* sb = snaps + (long long)b * 7 * acap;
    for (int i = tid; i < k; i += bs) {
        w[i] = w_io[(long long)b * K + i]; di[i] = 1.0 / dg[(long long)b * K + i]; ps[i] = pos[(long long)b * K + i];
    }
    __syncthreads();
    const double lb = lam[b], yb = yy[b], tb = tol[b], l2 = l2v[b];   // l2: elastic-net ridge term (augmented Lasso)
    int ep = 0;
    double gap_rel = 1e300;
    int ismax[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    while (true) {
        // ---- exact refresh q = c - G w (support only) ----
        if (tid == 0) { int na = 0; for (int j = 0; j < k; ++j) if (w[j] != 0.0) act[na++] = j; s_na = na; }
        __syncthreads();
        for (int i = tid; i < k; i += bs) q[i] = cb[i];
        __syncthreads();
        {
            const int na = s_na;
            for (int a = 0; a < na; ++a) {
                const int j = act[a];
                const GT* row = Gb + (long long)ps[j] * ldg;
                const double wj = w[j];
                for (int i = tid; i < k; i += bs) q[i] -= (double)row[ps[i]] * wj;
            }
        }
        __syncthreads();
        if (l2 != 0.0) { for (int i = tid; i < k; i += bs) q[i] -= l2 * w[i]; __syncthreads(); }
        // ---- gap ----
        {
            double v[4] = {0, 0, 0, 0};
            for (int i = tid; i < k; i += bs) { v[0] += cb[i] * w[i]; v[1] += q[i] * w[i]; v[2] += fabs(w[i]); v[3] = fmax(v[3], fabs(q[i])); }
            block_reduce(v, 4, red, tid, bs, ismax);
            if (tid == 0) {
                const double cw = red[0][0], qw = red[1][0], l1 = red[2][0], qm = red[3][0];
                const double rr = yb - cw - qw, ry = yb - cw;
                const double s = (qm > lb) ? lb / qm : 1.0;
                gap_rel = 2.0 * ((0.5 * rr + lb * l1) - (s * ry - 0.5 * s * s * rr)) / fmax(yb, 1e-300);
                s_stop = (gap_rel <= tb || ep >= max_ep) ? 1 : 0;
                red[0][0] = gap_rel;
            }
            __syncthreads();
            if (s_stop) { gap_rel = red[0][0]; break; }
        }
        // ---- full pass ----
        for (int j = 0; j < k; ++j) {
            if (tid == 0) {
                const double wj = w[j], z = wj + q[j] * di[j], th = lb * di[j];
                const double wn = (z > th) ? z - th : ((z < -th) ? z + th : 0.0);
                s_delta = wn - wj; w[j] = wn;
            }
            __syncthreads();
            const double dl = s_delta;
            if (dl != 0.0) {
                const GT* row = Gb + (long long)ps[j] * ldg;
                for (int i = tid; i < k; i += bs) q[i] -= ((double)row[ps[i]] + (i == j ? l2 : 0.0)) * dl;
            }
            __syncthreads();
        }
        ++ep;
        // ---- active phase ----
        if (tid == 0) { int na = 0; for (int j = 0; j < k; ++j) if (w[j] != 0.0) act[na++] = j; s_na = na; }
        __syncthreads();
        const int na = s_na;
        if (na == 0) continue;
        const bool compact = na <= acap && na <= MP * bs;
        if (compact) {
            for (int a = 0; a < na; ++a) {
                const GT* row = Gb + (long long)ps[act[a]] * ldg;
                for (int t = tid; t < na; t += bs) GAb[(long long)a * acap + t] = (float)((double)row[ps[act[t]]] + (t == a ? l2 : 0.0));
            }
            __syncthreads();
        }
        int nsnap = 0;
        for (int pass = 0; pass < act_passes && ep < max_ep; ++pass) {
            if (compact) {
                float cur[MP], nxt[MP];
#pragma unroll
                for (int m = 0; m < MP; ++m) { const int t = tid + m * bs; cur[m] = (t < na) ? GAb[t] : 0.f; }
                for (int a = 0; a < na; ++a) {
                    const int j = act[a];
                    if (tid == 0) {
                        const double wj = w[j], z = wj + q[j] * di[j], th = lb * di[j];
                        const double wn = (z > th) ? z - th : ((z < -th) ? z + th : 0.0);
                        s_delta = wn - wj; w[j] = wn;
                    }
                    if (a + 1 < na) {
                        const float* nr = GAb + (long long)(a + 1) * acap;
#pragma unroll
                        for (int m = 0; m < MP; ++m) { const int t = tid + m * bs; nxt[m] = (t < na) ? nr[t] : 0.f; }
                    }
                    __syncthreads();
                    const double dl = s_delta;
                    if (dl != 0.0) {
#pragma unroll
                        for (int m = 0; m < MP; ++m) { const int t = tid + m * bs; if (t < na) q[act[t]] -= (double)cur[m] * dl; }
                    }
                    __syncthreads();
#pragma unroll
                    for (int m = 0; m < MP; ++m) cur[m] = nxt[m];
                }
            } else {
                for (int a = 0; a < na; ++a) {
                    const int j = act[a];
                    if (tid == 0) {
                        const double wj = w[j], z = wj + q[j] * di[j], th = lb * di[j];
                        const double wn = (z > th) ? z - th : ((z < -th) ? z + th : 0.0);
                        s_delta = wn - wj; w[j] = wn;
                    }
                    __syncthreads();
                    const double dl = s_delta;
                    if (dl != 0.0) {
                        const GT* row = Gb + (long long)ps[j] * ldg;
                        for (int t = tid; t < na; t += bs) q[act[t]] -= ((double)row[ps[act[t]]] + (act[t] == j ? l2 : 0.0)) * dl;
                    }
                    __syncthreads();
                }
            }
            ++ep;
            if (anderson && compact && ((pass + 1) % snap_every == 0)) {
                for (int t = tid; t < na; t += bs) sb[nsnap * acap + t] = w[act[t]];
                ++nsnap;
                __syncthreads();
                if (nsnap == 5) {
                    nsnap = 0;
                    double acc[10];
                    for (int m = 0; m < 10; ++m) acc[m] = 0;
                    for (int t = tid; t < na; t += bs) {
                        double u[4];
                        for (int m = 0; m < 4; ++m) u[m] = sb[(m + 1) * acap + t] - sb[m * acap + t];
                        int h = 0;
                        for (int m = 0; m < 4; ++m) for (int l = m; l < 4; ++l) acc[h++] += u[m] * u[l];
                    }
                    int zm[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    block_reduce(acc, 10, red, tid, bs, zm);
                    if (tid == 0) {
                        double M[4][5]; int h = 0; double tr = 0;
                        for (int m = 0; m < 4; ++m) for (int l = m; l < 4; ++l) { M[m][l] = red[h][0]; M[l][m] = red[h][0]; ++h; }
                        for (int m = 0; m < 4; ++m) tr += M[m][m];
                        for (int m = 0; m < 4; ++m) { M[m][m] += 1e-10 * tr + 1e-300; M[m][4] = 1.0; }
                        int ok = tr > 0;
                        for (int m = 0; m < 4 && ok; ++m) {
                            int piv = m;
                            for (int l = m + 1; l < 4; ++l) if (fabs(M[l][m]) > fabs(M[piv][m])) piv = l;
                            if (fabs(M[piv][m]) < 1e-300) { ok = 0; break; }
                            for (int l = 0; l < 5; ++l) { double tmp = M[m][l]; M[m][l] = M[piv][l]; M[piv][l] = tmp; }
                            for (int l = m + 1; l < 4; ++l) { const double f = M[l][m] / M[m][m]; for (int g = m; g < 5; ++g) M[l][g] -= f * M[m][g]; }
                        }
                        double z[4], sz = 0;
                        for (int m = 3; m >= 0 && ok; --m) { double v = M[m][4]; for (int l = m + 1; l < 4; ++l) v -= M[m][l] * z[l]; z[m] = v / M[m][m]; sz += z[m]; }
                        if (ok && isfinite(sz) && fabs(sz) > 1e-300) { for (int m = 0; m < 4; ++m) s_coef[m] = z[m] / sz; } else ok = 0;
                        s_ok = ok;
                    }
                    __syncthreads();
                    if (s_ok) {
                        double* wa = sb + 5 * acap; double* qa = sb + 6 * acap;
                        for (int t = tid; t < na; t += bs) {
                            double v = 0;
                            for (int m = 0; m < 4; ++m) v += s_coef[m] * sb[(m + 1) * acap + t];
                            wa[t] = v; qa[t] = cb[act[t]];
                        }
                        __syncthreads();
                        for (int a = 0; a < na; ++a) {
                            const double wv = wa[a];
                            if (wv != 0.0) { const float* row = GAb + (long long)a * acap; for (int t = tid; t < na; t += bs) qa[t] -= (double)row[t] * wv; }
                        }
                        __syncthreads();
                        double v[6] = {0, 0, 0, 0, 0, 0};
                        for (int t = tid; t < na; t += bs) {
                            const int j = act[t];
                            v[0] += cb[j] * wa[t]; v[1] += qa[t] * wa[t]; v[2] += fabs(wa[t]);
                            v[3] += cb[j] * w[j]; v[4] += q[j] * w[j]; v[5] += fabs(w[j]);
                        }
                        int zm6[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                        block_reduce(v, 6, red, tid, bs, zm6);
                        if (tid == 0) {
                            const double Pa = 0.5 * (yb - red[0][0] - red[1][0]) + lb * red[2][0];
                            const double Pc = 0.5 * (yb - red[3][0] - red[4][0]) + lb * red[5][0];
                            s_ok = Pa < Pc;
                        }
                        __syncthreads();
                        if (s_ok) for (int t = tid; t < na; t += bs) { w[act[t]] = wa[t]; q[act[t]] = qa[t]; }
                        __syncthreads();
                    }
                }
            }
        }
    }
    for (int i = tid; i < k; i += bs) w_io[(long long)b * K + i] = w[i];
    if (tid == 0) { ep_out[b] = ep; gap_out[b] = gap_rel; }
}
"""

_MPS = (1, 2, 4, 8, 16, 32)
_kernel2_cache = {}


def _get_kernel2(gt, mp):
    import cupy as cp
    key = (cp.cuda.Device().id, gt)
    if key not in _kernel2_cache:
        names = ["cd2<%d>" % m for m in _MPS]
        mod = cp.RawModule(code="typedef %s GT;\n" % gt + _KERNEL2_SRC, options=("--std=c++14",), name_expressions=names)
        fs = {}
        lim = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"] - 8192
        for m, nm in zip(_MPS, names):
            fn = mod.get_function(nm)
            fn.max_dynamic_shared_size_bytes = lim
            fs[m] = fn
        _kernel2_cache[key] = fs
    return _kernel2_cache[key][mp]


def cd2_gram_batched(G, goff, ldg, pos, kk, c, dg, w, lam, yy, tol, max_ep=2000, act_passes=25, anderson=True,
                     acap=None, ga_bytes=2 ** 33, snap_every=5, l2=None):
    """Kernel v2 (see _KERNEL2_SRC). Same interface as cd_gram_batched; `acap` = compaction capacity."""
    import cupy as cp
    B, K = w.shape
    dev = w.device
    if B == 0:
        return torch.zeros(0, dtype=torch.int32, device=dev), torch.zeros(0, dtype=torch.float64, device=dev)
    if acap is None:
        acap = K
    # bound the compacted-Gram scratch memory
    while acap > 32 and B * acap * acap * 4 > ga_bytes:
        acap = int(acap * 0.8)
    bs = 128 if acap <= 2048 else 256
    mp = next((m for m in _MPS if m * bs >= acap), _MPS[-1])
    acap = min(acap, mp * bs)
    kern = _get_kernel2("double" if G.dtype == torch.float64 else "float", mp)
    smem = K * smem_per_k()
    ep = torch.empty(B, dtype=torch.int32, device=dev)
    gap = torch.empty(B, dtype=torch.float64, device=dev)
    snaps = torch.empty(B * 7 * acap, dtype=torch.float64, device=dev)
    GA = torch.empty(B * acap * acap, dtype=torch.float32, device=dev)
    a = cp.asarray
    torch.cuda.current_stream().synchronize()
    kern((B,), (bs,), (a(G), a(goff), np.int64(ldg), a(pos), a(kk), np.int32(K), a(c), a(dg), a(w),
                       a(lam), a(yy), a(tol), np.int32(max_ep), np.int32(act_passes), a(ep), a(gap),
                       np.int32(1 if anderson else 0), a(snaps), a(GA), np.int32(acap), np.int32(snap_every),
                       a(l2 if l2 is not None else torch.zeros(B, dtype=torch.float64, device=dev))),
         shared_mem=smem)
    cp.cuda.get_current_stream().synchronize()
    return ep, gap


class HostX:
    """X kept in pinned host memory as contiguous column blocks, streamed to the GPU on demand.

    For problems where X (n x p float32) does not fit next to the solver state on the GPU
    (p = 1e6, n = 1e4 is 40 GB). Blocks are prefetched on a side stream while the previous block's
    matmul runs. `cols(idx)` gathers arbitrary columns on the host (union growth)."""

    def __init__(self, X, block_bytes=2 ** 28):
        self.X = X
        self.n, self.p = X.shape
        self.shape = X.shape
        self.pc = max(1, int(block_bytes // (4 * self.n)))
        self.blocks = [torch.from_numpy(np.ascontiguousarray(X[:, j:j + self.pc], dtype=np.float32)).pin_memory()
                       for j in range(0, self.p, self.pc)]
        self.is_cuda = False
        self.bytes_streamed = 0

    def iter_blocks(self, dev):
        side = torch.cuda.Stream(dev)
        nxt = None
        with torch.cuda.stream(side):
            nxt = self.blocks[0].to(dev, non_blocking=True)
        for i in range(len(self.blocks)):
            torch.cuda.current_stream(dev).wait_stream(side)
            cur = nxt
            cur.record_stream(torch.cuda.current_stream(dev))
            if i + 1 < len(self.blocks):
                with torch.cuda.stream(side):
                    nxt = self.blocks[i + 1].to(dev, non_blocking=True)
            self.bytes_streamed += cur.numel() * 4
            yield i * self.pc, cur

    def cols(self, idx, dev):
        ii = idx.cpu().numpy()
        return torch.from_numpy(np.ascontiguousarray(self.X[:, ii], dtype=np.float32)).to(dev)


def full_corr_stats(Xd, Rt, M, thr=None, chunk_bytes=2 ** 31, inv_norm=None, lam=None, corr=None):
    """Stats of C = R^T X (B, p) computed in feature chunks.

    Xd: (n, p) float32 (on GPU, or a host tensor that is streamed chunk-wise if `stream=True`). Rt: (B, n) float32 residuals (already masked per fold).
    M: number of top |c_j| features to keep per problem.
    thr: optional (B,) thresholds; returns count of |c_j| > thr_b (strong rule).
    corr: optional (feature ids, problem idx, values): c[b, j] -= value before |.| (elastic net: the
        augmented-Lasso correlation is x_j^T r - l2 w_j on the support).
    inv_norm, lam: if given, candidates are ranked by the celer score (|c_j| - lam_b) / ||x_j||
    (distance to the dual constraint, valid for unnormalized columns) instead of |c_j|.
    Returns dict(qmax (B,) float64, top_idx (B, M) int64, top_val (B, M) float32, n_strong (B,)).
    """
    p = Xd.shape[1]
    B = Rt.shape[0]
    M = int(min(M, p))
    pc = max(M, int(chunk_bytes // (4 * max(B, 1))))
    pc = min(pc, p)
    dev = Rt.device
    qmax = torch.zeros(B, dtype=torch.float32, device=dev)
    nst = torch.zeros(B, dtype=torch.int64, device=dev) if thr is not None else None
    tv = ti = None
    thr32 = thr.to(torch.float32)[:, None] if thr is not None else None
    lam32 = lam.to(torch.float32)[:, None] if lam is not None else None
    if isinstance(Xd, HostX):
        it = Xd.iter_blocks(dev)
    else:
        it = ((j0, Xd[:, j0:j0 + pc]) for j0 in range(0, p, pc))
    for j0, Xc in it:
        Cc = Rt @ Xc                   # (B, pc)
        if corr is not None:
            cid, cb, cv = corr
            sel = (cid >= j0) & (cid < j0 + Cc.shape[1])
            if sel.any():
                Cc.index_put_((cb[sel], cid[sel] - j0), -cv[sel], accumulate=True)
        Cc.abs_()
        qmax = torch.maximum(qmax, Cc.max(dim=1).values)
        if thr is not None:
            nst += (Cc > thr32).sum(dim=1)
        if M > 0:
            Sc = Cc if inv_norm is None else (Cc - lam32) * inv_norm[j0:j0 + Cc.shape[1]][None, :]
            v, i = torch.topk(Sc, min(M, Sc.shape[1]), dim=1, sorted=False)
            i += j0
            if tv is None:
                tv, ti = v, i
            else:
                v2 = torch.cat([tv, v], 1)
                i2 = torch.cat([ti, i], 1)
                tv, sel = torch.topk(v2, M, dim=1, sorted=False)
                ti = torch.gather(i2, 1, sel)
        del Cc
    if tv is not None:
        tv, o = torch.sort(tv, dim=1, descending=True)
        ti = torch.gather(ti, 1, o)
    return dict(qmax=qmax.double(), top_idx=ti, top_val=tv, n_strong=nst)


def gap_from_stats(rr, ry, l1, qmax, lam, yy):
    """Relative duality gap 2*(P - D)/||y||^2 with dual point theta = r * min(1, lam/qmax)."""
    s = torch.clamp(lam / torch.clamp(qmax, min=1e-300), max=1.0)
    P = 0.5 * rr + lam * l1
    D = s * ry - 0.5 * s * s * rr
    return 2.0 * (P - D) / torch.clamp(yy, min=1e-300)


class Timer:
    """Accumulates wall time per phase with CUDA sync (for profiling where time goes)."""

    def __init__(self, sync=True):
        self.t = {}
        self.sync = sync
        self._stack = []

    def __call__(self, key):
        self._stack.append([key, None])
        return self

    def __enter__(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._stack[-1][1] = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        k, t0 = self._stack.pop()
        self.t[k] = self.t.get(k, 0.0) + time.perf_counter() - t0


def to_csc_list(rows, cols, vals, A, p, T):
    """Build A scipy CSC (p, T) matrices from per-alpha (feature, target, value) triplets."""
    import scipy.sparse as sp
    out = []
    for a in range(A):
        out.append(sp.csc_matrix((vals[a], (rows[a], cols[a])), shape=(p, T), dtype=np.float64))
    return out
