"""Gradient training of a readout on frozen cached maps, following dualneuron.training.trainer.

Kept from the reference: Adam lr 5e-3 (no weight decay, no clipping); batch 64, shuffle, drop_last; loss summed
over the batch and channels and scaled by sqrt(n_train / batch) plus gamma * sum|channel weights|;
ReduceLROnPlateau(mode=max, factor 0.3, patience 5, abs threshold 1e-6, min_lr 1e-4) on the validation metric,
stop after 3 LR decays or 100 epochs, restore the best-validation snapshot.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch


def poisson_sum(pred, target):
    return (pred - target * torch.log(pred.clamp(min=1e-3))).sum()


def mse_sum(pred, target):
    return 0.5 * ((pred - target) ** 2).sum()


def r2_cols(pred, target):
    """1 - SSE/SST per column (SST around the target's own mean), torch -> numpy."""
    sst = ((target - target.mean(0)) ** 2).sum(0)
    return (1 - ((target - pred) ** 2).sum(0) / sst.clamp(min=1e-12)).cpu().numpy()


def corr_cols(pred, target):
    p = pred - pred.mean(0)
    t = target - target.mean(0)
    return ((p * t).sum(0) / ((p.norm(dim=0) * t.norm(dim=0)).clamp(min=1e-12))).cpu().numpy()


@torch.no_grad()
def predict(model, X, bs=256, **kw):
    model.eval()
    return torch.cat([model(X[i:i + bs], **kw) for i in range(0, X.shape[0], bs)], 0)


def penalty(model, gamma, gamma_s=None):
    if hasattr(model, "l1_parts"):
        s, f = model.l1_parts()
        return gamma * f + (gamma if gamma_s is None else gamma_s) * s
    return gamma * model.l1()


def sparse_params(model):
    return [model.S, model.F] if hasattr(model, "l1_parts") else [model.features]


def _bcast(g, p):
    """Penalty g (float or per-channel (T,) tensor) broadcast against a parameter laid out (T, ...)."""
    if torch.is_tensor(g) and g.ndim == 1:
        return g.to(p.device).view(-1, *([1] * (p.ndim - 1)))
    return g


@torch.no_grad()
def prox_adam_(opt, params, gamma):
    """Preconditioned proximal step after an Adam step: w <- soft(w, lr * gamma / (sqrt(v_hat) + eps)).

    The L1 term is then not in the differentiated loss, so weights can reach (and stay at) exactly zero, which
    the reference's subgradient L1 under Adam never does. gamma: one value for all params, or a list (one per
    param); each value a float or a per-channel (T,) tensor (params laid out (T, ...)).
    """
    grp = opt.param_groups[0]
    lr, (_, b2), eps = grp["lr"], grp["betas"], grp["eps"]
    gams = gamma if isinstance(gamma, (list, tuple)) else [gamma] * len(params)
    for p, g in zip(params, gams):
        st = opt.state.get(p)
        if not st or (not torch.is_tensor(g) and g == 0):
            continue
        step = float(st["step"])
        denom = (st["exp_avg_sq"] / (1 - b2 ** step)).sqrt() + eps
        thr = lr * _bcast(g, p) / denom
        p.copy_(torch.sign(p) * (p.abs() - thr).clamp(min=0))


def roughness(model):
    """Per-channel scale-free roughness of the spatial masks: sum_r |grad S_tr|^2 / |S_tr|^2, shape (T,)."""
    C, H, W = model.shape
    S = model.S.view(model.T, model.rank, H, W)
    g2 = ((S[..., 1:, :] - S[..., :-1, :]) ** 2).sum((-1, -2)) + ((S[..., 1:] - S[..., :-1]) ** 2).sum((-1, -2))
    return (g2 / ((S ** 2).sum((-1, -2)) + 1e-12)).sum(1)


def train_member(model, Xtr, Ttr, Xva, Tva, loss="mse", gamma=0.0, gamma_s=None, metric="corr", lr=5e-3,
                 batch_size=64, max_epochs=100, lr_decay_factor=0.3, lr_decay_patience=5, lr_decay_steps=3,
                 lr_threshold=1e-6, min_lr=1e-4, seed=0, val_transform=None, log=None, l1_mode="subgrad",
                 nonneg_s=False, smooth_s=None, lr_schedule=None):
    """Train in place; returns dict(best_val, epochs, best_epoch, lr_hist, time_s). T* are loss targets.

    l1_mode: "subgrad" (reference: gamma * sum|w| added to the loss) or "prox" (prox_adam_ after each step).
    gamma / gamma_s: L1 weight of the features / spatial masks (gamma_s None = gamma); floats, or per-channel
      (T,) tensors in prox mode (channels are independent problems sharing only the LR schedule).
    nonneg_s: project factorized spatial masks onto S >= 0 after each step.
    smooth_s: weight (float or (T,) tensor) of the scale-free mask roughness penalty (`roughness`).
    lr_schedule: list of per-epoch learning rates; trains exactly len(lr_schedule) epochs with no validation
      (Xva / Tva unused, may be None) and keeps the final weights. Used to refit on all training rows with the
      schedule (and epoch count) a validated run chose.
    val_transform: optional callable mapping model output to the units of Tva (unused when None).
    """
    t0 = time.perf_counter()
    g = torch.Generator(device="cpu").manual_seed(seed)
    lossf = {"mse": mse_sum, "poisson": poisson_sum}[loss]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=lr_decay_factor,
                                                       patience=lr_decay_patience, threshold=lr_threshold,
                                                       threshold_mode="abs", min_lr=min_lr)
    n = Xtr.shape[0]
    scale = math.sqrt(n / batch_size)
    fixed = lr_schedule is not None
    gs = gamma if gamma_s is None else gamma_s
    best, best_state, n_decay, prev_lr, ep, best_ep, lr_hist = -np.inf, None, 0, lr, 0, 0, []
    n_ep = len(lr_schedule) if fixed else max_epochs
    for ep in range(1, n_ep + 1):
        if fixed:
            for grp in opt.param_groups:
                grp["lr"] = lr_schedule[ep - 1]
        lr_hist.append(opt.param_groups[0]["lr"])
        model.train()
        perm = torch.randperm(n, generator=g).to(Xtr.device)
        for i in range(0, n - batch_size + 1, batch_size):       # drop_last
            idx = perm[i:i + batch_size]
            out = model(Xtr[idx])
            obj = scale * lossf(out, Ttr[idx])
            if l1_mode == "subgrad":
                obj = obj + penalty(model, gamma, gamma_s)
            if smooth_s is not None:
                obj = obj + (_bcast(smooth_s, roughness(model)) * roughness(model)).sum()
            opt.zero_grad(set_to_none=True)
            obj.backward()
            opt.step()
            if l1_mode == "prox":
                if hasattr(model, "l1_parts"):
                    prox_adam_(opt, [model.S, model.F], [gs, gamma])
                elif not (not torch.is_tensor(gamma) and gamma == 0):
                    prox_adam_(opt, sparse_params(model), gamma)
            if nonneg_s:
                with torch.no_grad():
                    model.S.clamp_(min=0)
        if fixed:
            continue
        pv = predict(model, Xva)
        if val_transform is not None:
            pv = val_transform(pv)
        v = corr_cols(pv, Tva) if metric == "corr" else r2_cols(pv, Tva)
        v = float(np.nanmean(v))
        sched.step(v)
        cur = opt.param_groups[0]["lr"]
        if cur < prev_lr:
            n_decay, prev_lr = n_decay + 1, cur
        if v > best:
            best, best_state, best_ep = v, copy.deepcopy(model.state_dict()), ep
        if log is not None and (ep % 10 == 0 or ep == 1):
            log(f"    ep {ep:3d} val_{metric}={v:.4f} lr={cur:.1e}")
        if n_decay >= lr_decay_steps:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if fixed:
        best_ep = ep
    if Xtr.is_cuda:
        torch.cuda.synchronize()
    return dict(best_val=best, epochs=ep, best_epoch=best_ep, lr_hist=lr_hist, time_s=time.perf_counter() - t0)
