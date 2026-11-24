"""Train factorized readouts on a tier and save everything needed for selection / evaluation.

  python -m lassobench.factorized.run --tier unit-100k --model gauss_mse [--gammas 0,1,3,10] [--members 1,2,3,4,5]

Protocol (all models): 5 members, member m trains on a RandomState(m) 80% of the tier's training rows and is
early-stopped / scored on the other 20% (dualneuron's split_train_val). The test split is only predicted.
Per (penalty, member) we keep validation R2, test predictions (tier Y units), the pruned model's validation R2 and
test predictions, top-k truncated test R2, nonzero counts and the readout parameters. Selection (per-channel
penalty by mean member validation R2), ensembling and result rows are done by `summarize.py`.

"Pruned" = channel weights (and spatial-mask entries) with |w| < 1e-2 * max_c |w| of that channel set to 0, so
that nonzero counts of the subgradient-trained (never exactly sparse) models are honest; the pruned model is
what gets scored. ALS models are exactly sparse; pruning there is a no-op at the same threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from lassobench.factorized import als as A
from lassobench.factorized.data import OUT_DIR, load_maps, member_split
from lassobench.factorized.models import FactorizedReadout, GaussianReadout
from lassobench.factorized.train import predict, r2_cols, train_member

MODELS = {
    # (a) faithful dualneuron V4-resnet head: raw maps -> BN -> ReLU -> Gaussian readout -> ELU(x-1)+1, Poisson
    "gauss_poisson": dict(kind="gauss", input="raw", batchnorm=True, nonlin="relu", out="elu1", loss="poisson",
                          target="raw", metric="corr"),
    # (b) same head, linear output, MSE on the tier's z-scored Y
    "gauss_mse": dict(kind="gauss", input="raw", batchnorm=True, nonlin="relu", out="linear", loss="mse",
                      target="tier", metric="r2"),
    # (i) linear Gaussian point readout of the tier's X (exactly a constrained linear map of what Lasso sees)
    "gauss_lin": dict(kind="gauss", input="tier", batchnorm=False, nonlin="none", out="linear", loss="mse",
                      target="tier", metric="r2"),
    # same three heads with a proximal L1 step (exact zeros) instead of the reference's subgradient L1
    "gauss_poisson_prox": dict(kind="gauss", input="raw", batchnorm=True, nonlin="relu", out="elu1",
                               loss="poisson", target="raw", metric="corr", l1_mode="prox"),
    "gauss_mse_prox": dict(kind="gauss", input="raw", batchnorm=True, nonlin="relu", out="linear", loss="mse",
                           target="tier", metric="r2", l1_mode="prox"),
    "gauss_lin_prox": dict(kind="gauss", input="tier", batchnorm=False, nonlin="none", out="linear", loss="mse",
                           target="tier", metric="r2", l1_mode="prox"),
    # (ii)/(iii) space x feature factorized, gradient descent (Klindt-style random init), prox L1 on both factors
    "factgd_r1": dict(kind="fact", rank=1, input="tier", loss="mse", target="tier", metric="r2", l1_mode="prox"),
    "factgd_r2": dict(kind="fact", rank=2, input="tier", loss="mse", target="tier", metric="r2", l1_mode="prox"),
    "factgd_r3": dict(kind="fact", rank=3, input="tier", loss="mse", target="tier", metric="r2", l1_mode="prox"),
    # (ii)/(iii) same model by alternating Lasso solves, ridge-SVD init
    "als_r1": dict(kind="als", rank=1),
    "als_r2": dict(kind="als", rank=2),
    "als_r3": dict(kind="als", rank=3),
}
GAMMAS = [0, 1, 2, 3, 4.5, 6.5, 10, 15, 22, 30, 50, 100]
DEFAULT_GAMMAS = {k: GAMMAS for k in ("gauss_poisson", "gauss_mse", "gauss_lin", "gauss_poisson_prox",
                                       "gauss_mse_prox", "gauss_lin_prox", "factgd")}
ALS_FRAC_F = [0.6, 0.45, 0.35, 0.27, 0.2, 0.15, 0.11, 0.08, 0.06, 0.04]
ALS_FRAC_S = [0.05, 0.2, 0.5]
TOPK = [10, 30, 100, 300, 1000]
PRUNE_REL = 1e-2


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def prune_rel(w, dim, rel=PRUNE_REL):
    """Zero entries below rel * max|w| along `dim` (per channel)."""
    m = w.abs().amax(dim=dim, keepdim=True)
    return torch.where(w.abs() >= rel * m, w, torch.zeros_like(w))


def topk_mask(w, k, dim):
    """Keep the k largest |w| along dim."""
    k = min(k, w.shape[dim])
    idx = w.abs().topk(k, dim=dim).indices
    return torch.zeros_like(w).scatter(dim, idx, w.gather(dim, idx))


def lin_predict(Xflat, W, b):
    """Xflat (N, p), W (p, T) -> (N, T)."""
    return Xflat @ W + b


def run_gd(d, name, cfg, gammas, members, dev):
    C, H, Wd = d.shape
    T = d.Y.shape[1]
    raw_in = cfg["input"] == "raw"
    Xall = torch.as_tensor(np.ascontiguousarray(d.Xraw if raw_in else d.Xz), device=dev)
    Xte = torch.as_tensor(d.Xraw_test if raw_in else d.Xz_test, device=dev)
    Xte_flat_z = torch.as_tensor(d.Xz_test.reshape(len(d.Xz_test), -1), device=dev)
    Ytier = torch.as_tensor(d.Y, device=dev)
    if cfg["target"] == "raw":
        Ttgt = torch.as_tensor(np.clip(d.R, 0, None), device=dev)        # Poisson needs non-negative targets
        ymu = torch.as_tensor(d.y_mean, device=dev, dtype=torch.float32)
        ysd = torch.as_tensor(d.y_std, device=dev, dtype=torch.float32)
        to_tier = lambda p: (p - ymu) / ysd  # noqa: E731
        Tval_metric = torch.as_tensor(d.R, device=dev)
    else:
        Ttgt, to_tier, Tval_metric = Ytier, (lambda p: p), Ytier
    G, M, nte = len(gammas), len(members), Xte.shape[0]
    out = {k: [] for k in ("val_r2", "val_r2_pr", "test_pred", "test_pred_pr", "topk_r2", "nnz_ch", "eff_units",
                           "epochs", "time_s", "best_val")}
    params = {}
    linear_exact = cfg["kind"] == "fact" or (not cfg.get("batchnorm") and cfg.get("nonlin") == "none")
    for gi, gam in enumerate(gammas):
        for mi, seed in enumerate(members):
            tr, va = member_split(len(d.Y), seed)
            tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
            torch.manual_seed(seed)
            if cfg["kind"] == "gauss":
                model = GaussianReadout(d.shape, T, batchnorm=cfg["batchnorm"], nonlin=cfg["nonlin"],
                                        out=cfg["out"]).to(dev)
            else:
                model = FactorizedReadout(d.shape, T, rank=cfg["rank"], seed=seed).to(dev)
            info = train_member(model, Xall[tr_t], Ttgt[tr_t], Xall[va_t], Tval_metric[va_t], loss=cfg["loss"],
                                gamma=gam, metric=cfg["metric"], seed=seed, l1_mode=cfg.get("l1_mode", "subgrad"))
            Yva = Ytier[va_t]
            pv = to_tier(predict(model, Xall[va_t]))
            pt = to_tier(predict(model, Xte))
            if cfg["kind"] == "gauss":
                fpr = prune_rel(model.features.detach(), 0)
                pv_pr = to_tier(predict(model, Xall[va_t], features=fpr))
                pt_pr = to_tier(predict(model, Xte, features=fpr))
                nnz = (fpr != 0).sum(0)
                npos = (model.spatial_weights() > 1e-6).flatten(1).sum(1)
                eff = nnz * npos
                tk = []
                for k in TOPK:
                    pk = to_tier(predict(model, Xte, features=topk_mask(model.features.detach(), k, 0)))
                    tk.append(r2_cols(pk, torch.as_tensor(d.Y_test, device=dev)))
                    # topk here counts channel weights; effective units = k * npos (<= 4k)
                p_ = dict(features=model.features.detach().cpu().numpy(), mu=model.mu.detach().cpu().numpy(),
                          sigma=model.sigma.detach().cpu().numpy(),
                          bias=model.bias.detach().cpu().numpy())
            else:
                Spr, Fpr = prune_rel(model.S.detach(), 2), prune_rel(model.F.detach(), 2)
                pv_pr = predict(model, Xall[va_t], S=Spr, Fv=Fpr)
                pt_pr = predict(model, Xte, S=Spr, Fv=Fpr)
                Wpr = model.unit_weights(Spr, Fpr)
                nnz = (Fpr != 0).any(1).sum(1)                                # channels used by any rank
                eff = (Wpr != 0).reshape(-1, T).sum(0)
                Weff = model.unit_weights().reshape(-1, T)
                b = model.bias.detach()
                tk = [r2_cols(lin_predict(Xte_flat_z, topk_mask(Weff, k, 0), b),
                              torch.as_tensor(d.Y_test, device=dev)) for k in TOPK]
                p_ = dict(S=model.S.detach().cpu().numpy(), F=model.F.detach().cpu().numpy(),
                          bias=model.bias.detach().cpu().numpy())
            if linear_exact and cfg["kind"] == "gauss" and gi == 0 and mi == 0:
                # check: the eval readout equals the unit-basis linear map on the tier's X
                Wu = model.unit_weights().reshape(-1, T)
                diff = (lin_predict(Xte_flat_z, Wu, model.bias.detach()) - pt).abs().max().item()
                _log(f"  linear-equivalence check max|diff| = {diff:.2e}")
            for k_, v in p_.items():
                params.setdefault(k_, []).append(v)
            out["val_r2"].append(r2_cols(pv, Yva))
            out["val_r2_pr"].append(r2_cols(pv_pr, Yva))
            out["test_pred"].append(pt.cpu().numpy())
            out["test_pred_pr"].append(pt_pr.cpu().numpy())
            out["topk_r2"].append(np.stack(tk))
            out["nnz_ch"].append(nnz.cpu().numpy())
            out["eff_units"].append(eff.cpu().numpy())
            out["epochs"].append(info["epochs"])
            out["time_s"].append(info["time_s"])
            out["best_val"].append(info["best_val"])
            yte = torch.as_tensor(d.Y_test, device=dev)
            rel = d.reliable
            _log(f"{name} gamma={gam:g} seed={seed}: ep={info['epochs']} {info['time_s']:.1f}s "
                 f"val={info['best_val']:.3f} test R2 rel={r2_cols(pt, yte)[rel].mean():.3f} "
                 f"pruned={r2_cols(pt_pr, yte)[rel].mean():.3f} nnz_ch={nnz.float().mean():.1f} "
                 f"eff={eff.float().mean():.1f}")
    res = {k: np.asarray(v).reshape(G, M, *np.asarray(v[0]).shape) for k, v in out.items()}
    res.update({k: np.asarray(v).reshape(G, M, *np.asarray(v[0]).shape) for k, v in params.items()})
    res["penalties"] = np.asarray(gammas, float)
    return res


def run_als(d, name, cfg, members, dev, frac_f=ALS_FRAC_F, frac_s=ALS_FRAC_S, n_outer=6, iters=200):
    C, H, Wd = d.shape
    T, r = d.Y.shape[1], cfg["rank"]
    X = torch.as_tensor(d.Xz.reshape(len(d.Xz), C, H * Wd), device=dev)
    Xte = torch.as_tensor(d.Xz_test.reshape(len(d.Xz_test), C, H * Wd), device=dev)
    Y = torch.as_tensor(d.Y, device=dev)
    Yte = torch.as_tensor(d.Y_test, device=dev)
    pen = [(fs, ff) for fs in frac_s for ff in frac_f]
    G, M = len(pen), len(members)
    keys = ("val_r2", "val_r2_pr", "test_pred", "test_pred_pr", "topk_r2", "nnz_ch", "eff_units", "S", "F",
            "bias", "time_s")
    out = {k: [[None] * M for _ in range(G)] for k in keys}
    init = {k: [] for k in ("ridge_val_r2", "ridge_test_pred", "svd_val_r2", "svd_test_pred", "ridge_lambda")}
    lams = (np.logspace(1, 6, 11) * 1.0).tolist()
    for mi, seed in enumerate(members):
        tr, va = member_split(len(d.Y), seed)
        t0 = time.perf_counter()
        Xtr, Xva = X[tr], X[va]
        W, lam, r2r = A.ridge_dual(Xtr.reshape(len(tr), -1), Y[tr], Xva.reshape(len(va), -1), Y[va], lams)
        mx, my = Xtr.reshape(len(tr), -1).mean(0), Y[tr].mean(0)
        init["ridge_val_r2"].append(r2r.cpu().numpy())
        init["ridge_test_pred"].append(((Xte.reshape(len(Xte), -1) - mx) @ W + my).cpu().numpy())
        init["ridge_lambda"].append(lam.cpu().numpy())
        S0, F0 = A.svd_init(W, d.shape, r)
        b0 = my - torch.einsum("cs,trs,trc->t", mx.reshape(C, -1), S0, F0)
        init["svd_val_r2"].append(r2_cols(A.predict(Xva, S0, F0, b0), Y[va]))
        init["svd_test_pred"].append(A.predict(Xte, S0, F0, b0).cpu().numpy())
        t_init = time.perf_counter() - t0
        for si, fs in enumerate(frac_s):
            t1 = tp = time.perf_counter()
            for k, S, Fv, b in A.als_path(Xtr, Y[tr], S0, F0, frac_f, fs, n_outer=n_outer, iters=iters):
                dtp, tp = time.perf_counter() - tp, time.perf_counter()
                gi = si * len(frac_f) + k
                pv, pt = A.predict(Xva, S, Fv, b), A.predict(Xte, S, Fv, b)
                Spr, Fpr = prune_rel(S, 2), prune_rel(Fv, 2)
                Wpr = torch.einsum("trs,trc->cst", Spr, Fpr).reshape(-1, T)
                Weff = torch.einsum("trs,trc->cst", S, Fv).reshape(-1, T)
                o = out
                o["val_r2"][gi][mi] = r2_cols(pv, Y[va])
                o["val_r2_pr"][gi][mi] = r2_cols(A.predict(Xva, Spr, Fpr, b), Y[va])
                o["test_pred"][gi][mi] = pt.cpu().numpy()
                o["test_pred_pr"][gi][mi] = A.predict(Xte, Spr, Fpr, b).cpu().numpy()
                o["topk_r2"][gi][mi] = np.stack([r2_cols(Xte.reshape(len(Xte), -1) @ topk_mask(Weff, kk, 0) + b, Yte)
                                                 for kk in TOPK])
                o["nnz_ch"][gi][mi] = (Fpr != 0).any(1).sum(1).cpu().numpy()
                o["eff_units"][gi][mi] = (Wpr != 0).sum(0).cpu().numpy()
                o["S"][gi][mi], o["F"][gi][mi], o["bias"][gi][mi] = (S.cpu().numpy(), Fv.cpu().numpy(),
                                                                     b.cpu().numpy())
                o["time_s"][gi][mi] = 0.0
                rel = d.reliable
                _log(f"{name} seed={seed} frac_s={fs:g} frac_f={frac_f[k]:g}: test R2 rel="
                     f"{r2_cols(pt, Yte)[rel].mean():.3f} nnz_ch={o['nnz_ch'][gi][mi].mean():.0f} "
                     f"eff={o['eff_units'][gi][mi].mean():.1f} nnz_s={(S != 0).sum(2).float().mean():.1f} ({dtp:.1f}s)")
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            for k in range(len(frac_f)):   # path time spread over its points (+ the shared init time)
                out["time_s"][si * len(frac_f) + k][mi] = (dt + t_init / len(frac_s)) / len(frac_f)
        _log(f"{name} seed={seed}: ridge val R2 rel={init['ridge_val_r2'][-1][d.reliable].mean():.3f} "
             f"svd-r{r} val R2 rel={init['svd_val_r2'][-1][d.reliable].mean():.3f}")
    res = {k: np.asarray(v) for k, v in out.items()}
    res.update({k: np.asarray(v) for k, v in init.items()})
    res["penalties"] = np.asarray(pen, float)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--model", required=True, nargs="+", choices=list(MODELS))
    ap.add_argument("--gammas", default=None, help="comma list (gradient models)")
    ap.add_argument("--members", default="1,2,3,4,5")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [int(s) for s in a.members.split(",")]
    from lassobench.runner import git_info, hardware_info
    t0 = time.perf_counter()
    d = load_maps(a.tier, raw=any(MODELS[m].get("input") == "raw" for m in a.model))
    _log(f"loaded {a.tier}: maps {d.Xz.shape}, Y {d.Y.shape}, {time.perf_counter() - t0:.1f}s")
    os.makedirs(os.path.join(a.out_dir, a.tier), exist_ok=True)
    for name in a.model:
        cfg = MODELS[name]
        t1 = time.perf_counter()
        if cfg["kind"] == "als":
            res = run_als(d, name, cfg, members, dev)
        else:
            gk = "factgd" if cfg["kind"] == "fact" else name
            gammas = [float(g) for g in a.gammas.split(",")] if a.gammas else DEFAULT_GAMMAS[gk]
            res = run_gd(d, name, cfg, gammas, members, dev)
        wall = time.perf_counter() - t1
        hw = hardware_info()
        if dev.type == "cuda":
            hw["gpu_model_torch"] = torch.cuda.get_device_name()
            hw["gpu_model"] = hw.get("gpu_model") or hw["gpu_model_torch"]
        meta = dict(model=name, cfg=cfg, tier=a.tier, shape=d.shape, members=members, wall_s=wall,
                    hardware=hw, **git_info(), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    topk=TOPK, prune_rel=PRUNE_REL, argv=sys.argv)
        fp = os.path.join(a.out_dir, a.tier, f"{name}.npz")
        np.savez_compressed(fp, meta=json.dumps(meta, default=str), **res)
        _log(f"{name}: wall {wall:.1f}s -> {fp}")


if __name__ == "__main__":
    main()
