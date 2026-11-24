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


@torch.no_grad()
def prox_adam_(opt, params, gamma):
    """Preconditioned proximal step after an Adam step: w <- soft(w, lr * gamma / (sqrt(v_hat) + eps)).

    The L1 term is then not in the differentiated loss, so weights can reach (and stay at) exactly zero, which
    the reference's subgradient L1 under Adam never does.
    """
    grp = opt.param_groups[0]
    lr, (_, b2), eps = grp["lr"], grp["betas"], grp["eps"]
    for p in params:
        st = opt.state.get(p)
        if not st:
            continue
        step = float(st["step"])
        denom = (st["exp_avg_sq"] / (1 - b2 ** step)).sqrt() + eps
        thr = lr * gamma / denom
        p.copy_(torch.sign(p) * (p.abs() - thr).clamp(min=0))


def train_member(model, Xtr, Ttr, Xva, Tva, loss="mse", gamma=0.0, gamma_s=None, metric="corr", lr=5e-3,
                 batch_size=64, max_epochs=100, lr_decay_factor=0.3, lr_decay_patience=5, lr_decay_steps=3,
                 lr_threshold=1e-6, min_lr=1e-4, seed=0, val_transform=None, log=None, l1_mode="subgrad"):
    """Train in place; returns dict(best_val, epochs, time_s). T* are loss targets; val metric on Tva.

    l1_mode: "subgrad" (reference: gamma * sum|w| added to the loss) or "prox" (prox_adam_ after each step).
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
    best, best_state, n_decay, prev_lr, ep = -np.inf, None, 0, lr, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(Xtr.device)
        for i in range(0, n - batch_size + 1, batch_size):       # drop_last
            idx = perm[i:i + batch_size]
            out = model(Xtr[idx])
            obj = scale * lossf(out, Ttr[idx])
            if l1_mode == "subgrad":
                obj = obj + penalty(model, gamma, gamma_s)
            opt.zero_grad(set_to_none=True)
            obj.backward()
            opt.step()
            if l1_mode == "prox" and gamma > 0:
                prox_adam_(opt, sparse_params(model), gamma)
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
            best, best_state = v, copy.deepcopy(model.state_dict())
        if log is not None and (ep % 10 == 0 or ep == 1):
            log(f"    ep {ep:3d} val_{metric}={v:.4f} lr={cur:.1e}")
        if n_decay >= lr_decay_steps:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if Xtr.is_cuda:
        torch.cuda.synchronize()
    return dict(best_val=best, epochs=ep, time_s=time.perf_counter() - t0)
