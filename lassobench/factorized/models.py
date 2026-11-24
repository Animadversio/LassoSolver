"""Factorized readouts (reimplemented from dualneuron / neuralpredictors, not imported).

GaussianReadout  : dualneuron FullGaussian2d (gauss_type='isotropic', as the V4 twins use) behind an optional
                   BatchNorm2d -> ReLU front end and an optional ELU(x-1)+1 output.
FactorizedReadout: W_t = sum_r S_{t,r} (H x W spatial mask) outer F_{t,r} (C feature vector), Klindt et al. 2017.

Both read a (N, C, H, W) map and return (N, T). Both can export their eval-mode readout as an equivalent
unit-basis weight tensor (C, H, W, T) when the front end is the identity (linear variants).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianReadout(nn.Module):
    """Isotropic Gaussian point readout, as dualneuron.twins.layers.FullGaussian2d(gauss_type='isotropic').

    mu (T, 2) in [-1, 1] grid coordinates (x = width, y = height, align_corners=True), sigma (T,) >= 0.
    Train mode: per batch element a position ~ N(mu, sigma^2 I), clamped to [-1, 1]; eval: position = mu.
    Channel weights `features` (C, T) init 1/C; bias (T,) init 0.
    """

    def __init__(self, shape, T, bias=True, init_mu_range=0.4, init_sigma=0.6, batchnorm=True, nonlin="relu",
                 out="elu1", elu_offset=-1.0, bn_momentum=0.1):
        super().__init__()
        C, H, W = shape
        self.shape, self.T, self.out, self.elu_offset = tuple(shape), T, out, elu_offset
        self.bn = nn.BatchNorm2d(C, momentum=bn_momentum) if batchnorm else None
        self.nonlin = {"relu": nn.ReLU(), "gelu": nn.GELU(), "none": nn.Identity()}[nonlin]
        self.mu = nn.Parameter(torch.empty(T, 2).uniform_(-init_mu_range, init_mu_range))
        self.sigma = nn.Parameter(torch.full((T,), float(init_sigma)))
        self.features = nn.Parameter(torch.full((C, T), 1.0 / C))
        self.bias = nn.Parameter(torch.zeros(T)) if bias else None

    def front(self, x):
        if self.bn is not None:
            x = self.bn(x)
        return self.nonlin(x)

    def grid(self, N, sample):
        with torch.no_grad():
            self.mu.clamp_(-1, 1)
            self.sigma.clamp_(min=0)
        g = self.mu[None].expand(N, -1, -1)
        if sample:
            g = g + torch.randn_like(g) * self.sigma[None, :, None]
        return g.clamp(-1, 1)[:, :, None, :]            # (N, T, 1, 2)

    def readout(self, x, features=None, sample=None):
        sample = self.training if sample is None else sample
        g = self.grid(x.shape[0], sample)
        v = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=True)  # (N, C, T, 1)
        f = self.features if features is None else features
        y = (v[..., 0] * f[None]).sum(1)
        return y + self.bias if self.bias is not None else y

    def forward(self, x, features=None, sample=None):
        y = self.readout(self.front(x), features, sample)
        return F.elu(y + self.elu_offset) + 1 if self.out == "elu1" else y

    def l1(self):
        return self.features.abs().sum()

    def sparse_param(self):
        return self.features                           # (C, T): what L1 and pruning act on

    @torch.no_grad()
    def spatial_weights(self):
        """Eval-mode bilinear weights (T, H, W) of the read position (align_corners=True)."""
        C, H, W = self.shape
        mu = self.mu.clamp(-1, 1)
        px = (mu[:, 0] + 1) / 2 * (W - 1)
        py = (mu[:, 1] + 1) / 2 * (H - 1)
        wx = (1 - (torch.arange(W, device=mu.device)[None] - px[:, None]).abs()).clamp(min=0)   # (T, W)
        wy = (1 - (torch.arange(H, device=mu.device)[None] - py[:, None]).abs()).clamp(min=0)   # (T, H)
        return wy[:, :, None] * wx[:, None, :]

    @torch.no_grad()
    def unit_weights(self, features=None):
        """(C, H, W, T) unit-basis weights of the eval readout (exact only when the front end is identity)."""
        f = self.features if features is None else features
        return torch.einsum("ct,thw->chwt", f, self.spatial_weights())

    @torch.no_grad()
    def mu_pixels(self):
        C, H, W = self.shape
        mu = self.mu.clamp(-1, 1)
        return torch.stack([(mu[:, 1] + 1) / 2 * (H - 1), (mu[:, 0] + 1) / 2 * (W - 1)], 1)  # (T, 2) = (row, col)


class FactorizedReadout(nn.Module):
    """Rank-r space x feature factorized linear readout: y_t = sum_r <S_{t,r} (x) F_{t,r}, x> + b_t.

    S: (T, r, H*W) spatial masks, F: (T, r, C) feature vectors. Penalty lam_s |S|_1 + lam_f |F|_1.
    """

    def __init__(self, shape, T, rank=1, bias=True, init="random", S0=None, F0=None, seed=None):
        super().__init__()
        C, H, W = shape
        self.shape, self.T, self.rank = tuple(shape), T, rank
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        if S0 is not None:
            S, Fv = torch.as_tensor(S0, dtype=torch.float32), torch.as_tensor(F0, dtype=torch.float32)
        else:   # Klindt-style: small positive-ish random mask, random features
            S = torch.rand(T, rank, H * W, generator=g) / (H * W)
            Fv = torch.randn(T, rank, C, generator=g) / C ** 0.5
        self.S = nn.Parameter(S.clone())
        self.F = nn.Parameter(Fv.clone())
        self.bias = nn.Parameter(torch.zeros(T)) if bias else None
        self.out = "linear"

    def forward(self, x, S=None, Fv=None, sample=None):
        S = self.S if S is None else S
        Fv = self.F if Fv is None else Fv
        N, C, H, W = x.shape
        xs = x.reshape(N, C, H * W)
        # z[n, t, r, c] = sum_s x[n, c, s] S[t, r, s]  -> then contract with F over c
        z = torch.einsum("ncs,trs->ntrc", xs, S)
        y = torch.einsum("ntrc,trc->nt", z, Fv)
        return y + self.bias if self.bias is not None else y

    def l1_parts(self):
        return self.S.abs().sum(), self.F.abs().sum()

    @torch.no_grad()
    def unit_weights(self, S=None, Fv=None):
        S = self.S if S is None else S
        Fv = self.F if Fv is None else Fv
        C, H, W = self.shape
        return torch.einsum("trs,trc->cst", S, Fv).reshape(C, H, W, self.T)
