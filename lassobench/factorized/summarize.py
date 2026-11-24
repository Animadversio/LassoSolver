"""Selection, ensembling, stability and result rows for the factorized-readout runs saved by `run.py`.

  python -m lassobench.factorized.summarize --tier unit-100k [--write-rows]

Selection: for each channel the penalty with the best mean (over the 5 members) validation R2 of the *pruned*
model; the reported model is the 5-member ensemble (prediction average) at that penalty. Validation rows are
the members' own 20% held-out training rows; the test split is never used for any choice.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import uuid

import numpy as np

from lassobench.factorized.data import OUT_DIR, load_maps

MODEL_ORDER = ["gauss_poisson", "gauss_mse", "gauss_lin", "gauss_poisson_prox", "gauss_mse_prox", "gauss_lin_prox",
               "factgd_r1", "factgd_r2", "factgd_r3", "als_r1", "als_r2",
               "als_r3"]
LOSS = {"gauss_poisson": "Poisson NLL, ELU(x-1)+1", "gauss_poisson_prox": "Poisson NLL, ELU(x-1)+1"}


def r2(Yt, P):
    Yt = np.asarray(Yt, np.float64)
    return 1 - ((Yt - P) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)


def corr(Yt, P):
    a, b = Yt - Yt.mean(0), P - P.mean(0)
    return (a * b).sum(0) / np.maximum(np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0), 1e-12)


def prune_np(w, axis, rel=1e-2):
    m = np.abs(w).max(axis=axis, keepdims=True)
    return np.where(np.abs(w) >= rel * m, w, 0)


def bilinear_np(mu, H, W):
    """mu (T, 2) grid coords (x, y) -> (T, H, W) bilinear weights (align_corners=True)."""
    mu = np.clip(mu, -1, 1)
    px, py = (mu[:, 0] + 1) / 2 * (W - 1), (mu[:, 1] + 1) / 2 * (H - 1)
    wx = np.clip(1 - np.abs(np.arange(W)[None] - px[:, None]), 0, None)
    wy = np.clip(1 - np.abs(np.arange(H)[None] - py[:, None]), 0, None)
    return wy[:, :, None] * wx[:, None, :]


def mu_px(mu, H, W):
    mu = np.clip(mu, -1, 1)
    return np.stack([(mu[..., 1] + 1) / 2 * (H - 1), (mu[..., 0] + 1) / 2 * (W - 1)], -1)  # (row, col)


def unit_weights(z, kind, g, m, shape, prune=True):
    """(C, H, W, T) unit-basis weight of member m at penalty g (pruned as scored). Gauss: linearized readout."""
    C, H, W = shape
    if kind == "gauss":
        f = z["features"][g, m]
        f = prune_np(f, 0) if prune else f
        return np.einsum("ct,thw->chwt", f, bilinear_np(z["mu"][g, m], H, W))
    S, F = z["S"][g, m], z["F"][g, m]
    if prune:
        S, F = prune_np(S, 2), prune_np(F, 2)
    return np.einsum("trs,trc->cst", S, F).reshape(C, H, W, -1)


def peak_px(Wu):
    """(C, H, W, T) -> (T, 2) position (row, col) of max sum_c |w|."""
    mass = np.abs(Wu).sum(0)
    H, W, T = mass.shape
    k = mass.reshape(H * W, T).argmax(0)
    return np.stack([k // W, k % W], 1).astype(float)


def model_kind(name):
    return "gauss" if name.startswith("gauss") else "fact"


def summarize_model(name, z, d):
    meta = json.loads(str(z["meta"]))
    shape = tuple(meta["shape"])
    C, H, W = shape
    kind = model_kind(name)
    rel = d.reliable
    Yte = d.Y_test.astype(np.float64)
    val = np.nanmean(z["val_r2_pr"], 1)                      # (G, T)
    val = np.where(np.isfinite(val), val, -np.inf)
    G, M, T = z["val_r2_pr"].shape
    gsel = val.argmax(0)                                     # per-channel penalty
    tt = np.arange(T)
    P_ens = z["test_pred_pr"][gsel, :, :, tt].mean(1).T      # (n_test, T)   [fancy index -> (T, M, n)]
    P_ens_unpr = z["test_pred"][gsel, :, :, tt].mean(1).T
    r2_ens, c_ens = r2(Yte, P_ens), corr(Yte, P_ens)
    r2_unpr = r2(Yte, P_ens_unpr)
    r2_mem = np.stack([r2(Yte, z["test_pred_pr"][gsel, m, :, tt].T) for m in range(M)])  # (M, T)
    eff = z["eff_units"][gsel, :, tt].astype(float)          # (T, M)
    nch = z["nnz_ch"][gsel, :, tt].astype(float)
    # per-penalty (global) frontier: ensemble pruned R2 vs member-mean effective units
    frontier = []
    for g in range(G):
        Pg = z["test_pred_pr"][g].mean(0)
        frontier.append(dict(penalty=np.asarray(z["penalties"][g]).tolist(),
                             r2_rel=float(r2(Yte, Pg)[rel].mean()),
                             r2_member_rel=float(np.mean([r2(Yte, z["test_pred_pr"][g, m])[rel].mean()
                                                          for m in range(M)])),
                             eff_units_rel=float(z["eff_units"][g][:, rel].mean()),
                             nnz_ch_rel=float(z["nnz_ch"][g][:, rel].mean()),
                             val_r2_mean=float(np.nanmean(z["val_r2_pr"][g]))))
    gglob = int(np.argmax([np.nanmean(z["val_r2_pr"][g]) for g in range(G)]))
    topk = z["topk_r2"][gsel, :, :, tt].mean(1)              # (T, K)
    # union support of the ensemble + stability across members
    union = np.zeros(T)
    jac, cos, pos_d = [], [], []
    Wsel = np.zeros((M, C * H * W, T), np.float32)           # member weights at each channel's selected penalty
    for g in np.unique(gsel):
        cols = np.flatnonzero(gsel == g)
        for m in range(M):
            Wsel[m][:, cols] = unit_weights(z, kind, g, m, shape).reshape(-1, T)[:, cols]
    for t in range(T):
        sup = [set(np.flatnonzero(Wsel[m, :, t])) for m in range(M)]
        if kind == "gauss":
            vecs = [prune_np(z["features"][gsel[t], m][:, t], 0) for m in range(M)]
            pos = [mu_px(z["mu"][gsel[t], m][t], H, W) for m in range(M)]
        else:
            vecs = [Wsel[m, :, t] for m in range(M)]
            pos = [peak_px(Wsel[m, :, t].reshape(C, H, W, 1))[0] for m in range(M)]
        union[t] = len(set().union(*sup))
        js, cs, ds = [], [], []
        for a, b in itertools.combinations(range(M), 2):
            u = len(sup[a] | sup[b])
            js.append(len(sup[a] & sup[b]) / u if u else 1.0)
            na, nb = np.linalg.norm(vecs[a]), np.linalg.norm(vecs[b])
            cs.append(float(vecs[a] @ vecs[b] / (na * nb)) if na > 0 and nb > 0 else np.nan)
            ds.append(float(np.linalg.norm(pos[a] - pos[b])))
        jac.append(np.mean(js)); cos.append(np.nanmean(cs)); pos_d.append(np.mean(ds))
    jac, cos, pos_d = map(np.asarray, (jac, cos, pos_d))
    if kind == "gauss":
        nparam = C + 4 + (2 * C / T if meta["cfg"].get("batchnorm") else 0)
    else:
        r_ = meta["cfg"]["rank"]
        nparam = r_ * (C + H * W) + 1
    tsum = float(np.sum(z["time_s"]))
    m = dict(r2_test_mean=float(r2_ens.mean()), r2_test_median=float(np.median(r2_ens)),
             r2_test_mean_reliable=float(r2_ens[rel].mean()), r2_test_median_reliable=float(np.median(r2_ens[rel])),
             corr_test_mean_reliable=float(c_ens[rel].mean()), corr_test_mean=float(c_ens.mean()),
             r2_test_mean_reliable_unpruned=float(r2_unpr[rel].mean()),
             r2_test_mean_reliable_single_member=float(r2_mem[:, rel].mean()),
             r2_test_mean_reliable_single_member_sd=float(r2_mem[:, rel].mean(1).std()),
             r2_test_mean_reliable_global_penalty=float(r2(Yte, z["test_pred_pr"][gglob].mean(0))[rel].mean()),
             global_penalty=np.asarray(z["penalties"][gglob]).tolist(),
             nnz_mean=float(eff.mean()), nnz_median=float(np.median(eff.mean(1))),
             nnz_mean_reliable=float(eff[rel].mean()), nnz_union_mean_reliable=float(union[rel].mean()),
             nnz_union_mean=float(union.mean()),
             nnz_channels_mean_reliable=float(nch[rel].mean()), nnz_channels_mean=float(nch.mean()),
             n_params_per_channel=float(nparam), n_reliable=int(rel.sum()),
             stab_jaccard_mean_reliable=float(jac[rel].mean()), stab_jaccard_median_reliable=float(np.median(jac[rel])),
             stab_cosine_mean_reliable=float(np.nanmean(cos[rel])),
             stab_position_dist_px_mean_reliable=float(pos_d[rel].mean()),
             stab_position_dist_px_median_reliable=float(np.median(pos_d[rel])),
             topk=meta["topk"], topk_r2_mean_reliable=topk[rel].mean(0).tolist(),
             penalty_selected_hist=np.bincount(gsel, minlength=G).tolist(),
             time_member_fit_sum_s=tsum, frontier=frontier)
    if "ridge_test_pred" in z.files:
        m["ridge80_ens_r2_test_mean_reliable"] = float(r2(Yte, z["ridge_test_pred"].mean(0))[rel].mean())
        m["svd_init_ens_r2_test_mean_reliable"] = float(r2(Yte, z["svd_test_pred"].mean(0))[rel].mean())
    per = dict(r2=r2_ens, corr=c_ens, gsel=gsel, eff=eff.mean(1), union=union, jac=jac, cos=cos, pos_d=pos_d,
               r2_mem=r2_mem)
    return meta, m, per


def to_row(name, meta, m, tier_shape, raw_path):
    return dict(run_id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), task="CV", solver=f"fact_{name}", tier=meta["tier"],
                shape=tier_shape, max_targets=None, tag="factorized-readout",
                config=dict(device="cuda", model=meta["cfg"], loss=LOSS.get(name, "MSE (tier Y), linear"),
                            penalties=None, members=meta["members"], prune_rel=meta["prune_rel"],
                            selection="per-channel penalty by mean member val R2 of the pruned model; "
                                      "5-member ensemble (each member: 80% of train rows, 20% val)",
                            nnz_definition="effective (c,h,w) unit-basis weights per channel, member mean"),
                hardware=meta["hardware"], git_hash=meta["git_hash"], git_dirty=meta["git_dirty"], status="ok",
                time_total_s=meta["wall_s"], time_prepare_s=0.0, time_solve_s=meta["wall_s"], metrics=m,
                raw=raw_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--write-rows", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    d = load_maps(a.tier, raw=False)
    tier_shape = dict(n_train=int(d.Y.shape[0]), n_test=int(d.Y_test.shape[0]), p=int(d.p), T=int(d.Y.shape[1]))
    summ = {}
    for name in a.models or MODEL_ORDER:
        fp = os.path.join(a.out_dir, a.tier, f"{name}.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=False)
        meta, m, per = summarize_model(name, z, d)
        summ[name] = dict(meta=meta, metrics=m)
        np.savez_compressed(os.path.join(a.out_dir, a.tier, f"{name}_summary.npz"), **per)
        gpu = meta["hardware"].get("gpu_model")
        print(f"{name:14s} R2rel={m['r2_test_mean_reliable']:.3f} (unpr {m['r2_test_mean_reliable_unpruned']:.3f},"
              f" member {m['r2_test_mean_reliable_single_member']:.3f}, glob {m['r2_test_mean_reliable_global_penalty']:.3f})"
              f" corr={m['corr_test_mean_reliable']:.3f} eff={m['nnz_mean_reliable']:.0f} union={m['nnz_union_mean_reliable']:.0f}"
              f" nch={m['nnz_channels_mean_reliable']:.0f} jac={m['stab_jaccard_mean_reliable']:.2f}"
              f" cos={m['stab_cosine_mean_reliable']:.2f} dpos={m['stab_position_dist_px_mean_reliable']:.2f}px"
              f" wall={meta['wall_s']:.0f}s {gpu}", flush=True)
        if a.write_rows:
            from lassobench.runner import append_row
            append_row(to_row(name, meta, m, tier_shape, fp))
    with open(os.path.join(a.out_dir, a.tier, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)


if __name__ == "__main__":
    main()
