#!/usr/bin/env python
"""factorized-readout figures + mu-vs-Lasso-position analysis (CPU, reads STORE/results/factorized/<tier>/).

  python scripts/dev_fact_figures.py --tier unit-100k

Writes figures/factorized/<tier>_frontier.png, <tier>_concentration.png, <tier>_mu_vs_lasso.png and
STORE/results/factorized/<tier>/mu_vs_lasso.json.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from lassobench.factorized.data import OUT_DIR, load_maps  # noqa: E402
from lassobench.factorized.summarize import MODEL_ORDER, mu_px, peak_px, unit_weights  # noqa: E402
from lassobench.runner import read_results  # noqa: E402

FIG = os.path.join(REPO, "figures", "factorized")
COL = {"gauss_poisson": "#d62728", "gauss_mse": "#ff7f0e", "gauss_lin": "#bcbd22", "gauss_poisson_prox": "#f7b6d2",
       "gauss_mse_prox": "#ffbb78", "gauss_lin_prox": "#dbdb8d", "factgd_r1": "#9467bd",
       "factgd_r2": "#8c564b", "factgd_r3": "#e377c2", "als_r1": "#1f77b4", "als_r2": "#17becf", "als_r3": "#2ca02c"}


def ridge_ref(tier):
    rows = [r for r in read_results() if r.get("tier") == tier and r.get("solver") == "cpu_ridgecv"
            and r.get("status") == "ok" and r.get("task") == "CV"]
    return rows[-1]["metrics"]["r2_test_mean_reliable"] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="unit-100k")
    a = ap.parse_args()
    od = os.path.join(OUT_DIR, a.tier)
    os.makedirs(FIG, exist_ok=True)
    d = load_maps(a.tier, raw=False)
    C, H, W = d.shape
    rel = d.reliable
    summ = json.load(open(os.path.join(od, "summary.json")))
    lz = np.load(os.path.join(od, "lasso.npz")) if os.path.exists(os.path.join(od, "lasso.npz")) else None
    ls = json.load(open(os.path.join(od, "lasso_summary.json"))) if lz is not None else None
    rr = ridge_ref(a.tier)

    # --- frontier: R2 (reliable mean) vs effective units, and vs nonzero channel weights
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, xl in ((axs[0], "eff_units_rel", "effective nonzero (c,h,w) units per channel"),
                        (axs[1], "nnz_ch_rel", "nonzero feature-channel weights per channel")):
        if lz is not None:
            pn = lz["path_nnz"][:, rel].mean(1)
            pr = lz["path_r2"][:, rel].mean(1)
            ax.plot(np.maximum(pn, 0.8), pr, "k.-", lw=1, ms=3, label="Lasso path (unit basis, all train)")
            if "cv_r2" in lz.files:
                ax.plot(ls["cv"]["nnz_rel"], ls["cv"]["r2_rel"], "k*", ms=14, label="Lasso CV (harness)")
            m = ls["members"]
            ax.plot(m["nnz_mean_reliable"], m["r2_test_mean_reliable"], "kP", ms=11,
                    label="Lasso 5-member ens.")
        for name in MODEL_ORDER:
            if name not in summ:
                continue
            fr = summ[name]["metrics"]["frontier"]
            grp = [f["penalty"][0] if isinstance(f["penalty"], list) else 0 for f in fr]   # ALS: one line per frac_s
            for gv, ls_ in zip(sorted(set(grp)), ["-", "--", ":"]):
                sel = [f for f, gg in zip(fr, grp) if gg == gv]
                x = np.array([f[key] for f in sel])
                y = np.array([f["r2_rel"] for f in sel])
                o = np.argsort(x)
                ax.plot(np.maximum(x[o], 0.8), y[o], ".", ls=ls_, color=COL[name], lw=0.8, ms=4, alpha=0.6)
            mm = summ[name]["metrics"]
            xs = mm["nnz_mean_reliable"] if key == "eff_units_rel" else mm["nnz_channels_mean_reliable"]
            ax.plot(xs, mm["r2_test_mean_reliable"], "o", color=COL[name], ms=9, mec="k", label=f"fact_{name}")
        if rr is not None:
            ax.axhline(rr, color="gray", ls="--", label=f"RidgeCV {rr:.3f} (dense)")
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("test R2, mean over reliable channels")
        ax.set_ylim(bottom=max(-0.05, ax.get_ylim()[0]))
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, loc="lower right")
    axs[1].set_title("(Lasso: 1 unit = 1 channel weight)", fontsize=9)
    fig.suptitle(f"{a.tier} ({d.meta['layer']}, {C}x{H}x{W}): R2 vs sparsity. Lines = penalty sweep (ensemble of "
                 f"5, pruned); big markers = per-channel selected penalty", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{a.tier}_frontier.png"), dpi=130)

    if lz is None:
        return
    # --- concentration of Lasso weights
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    f1, f4, n1, n4 = (lz[f"conc_cv_{k}"] for k in ("frac1", "frac4", "null1", "null4"))
    ax = axs[0]
    idx = np.flatnonzero(rel)
    ax.scatter(n1[idx], f1[idx], c="C0", label="best single position")
    ax.scatter(n4[idx], f4[idx], c="C1", label="best 2x2 window")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("null: same units, random positions")
    ax.set_ylabel("Lasso |w| mass fraction")
    ax.set_title("Lasso (harness CV) spatial concentration, reliable ch.", fontsize=9)
    ax.legend(fontsize=8)
    ax = axs[1]
    mass = np.abs(lz["cv_coef"]).reshape(C, H, W, -1).sum(0)
    tot = mass.sum((0, 1))
    pooled = (mass[..., idx] / np.maximum(tot[idx], 1e-12)).mean(-1)
    im = ax.imshow(pooled, cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("mean over reliable ch. of per-position |w| mass fraction", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{a.tier}_concentration.png"), dpi=130)

    # --- Gaussian mu vs Lasso peak / centre of mass
    peak, com = lz["conc_cv_peak"].astype(float), lz["conc_cv_com"]
    res = {}
    gauss = [n for n in ("gauss_poisson", "gauss_mse", "gauss_lin", "gauss_mse_prox", "gauss_lin_prox") if n in summ]
    fig, axs = plt.subplots(1, max(1, len(gauss)), figsize=(4.5 * max(1, len(gauss)), 4.3), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, name in zip(axs[0], gauss):
        z = np.load(os.path.join(od, f"{name}.npz"))
        s = np.load(os.path.join(od, f"{name}_summary.npz"))
        gsel = s["gsel"]
        T = len(gsel)
        mu = np.stack([mu_px(z["mu"][gsel[t], :, t], H, W).mean(0) for t in range(T)])   # member-mean (row, col)
        dpk = np.linalg.norm(mu - peak, axis=1)
        dcm = np.linalg.norm(mu - com, axis=1)
        # null: Lasso peak of a random other reliable channel
        null = np.mean([np.linalg.norm(mu[idx] - peak[rng.permutation(idx)], axis=1).mean() for _ in range(500)])
        unif = np.mean([np.linalg.norm(mu[idx] - rng.uniform(0, [H - 1, W - 1], (len(idx), 2)), axis=1).mean()
                        for _ in range(500)])
        null_cm = np.mean([np.linalg.norm(mu[idx] - com[rng.permutation(idx)], axis=1).mean() for _ in range(500)])
        res[name] = dict(dist_mu_lassopeak_px_mean_rel=float(dpk[idx].mean()), null_com_shuffled_px=float(null_cm),
                         dist_mu_lassopeak_px_median_rel=float(np.median(dpk[idx])),
                         dist_mu_lassocom_px_mean_rel=float(dcm[idx].mean()),
                         null_shuffled_channels_px=float(null), null_uniform_px=float(unif),
                         frac_within_1px_rel=float((dpk[idx] <= 1.0).mean()))
        ax.scatter(peak[idx, 1], peak[idx, 0], marker="s", s=60, facecolor="none", edgecolor="k", label="Lasso peak")
        ax.scatter(mu[idx, 1], mu[idx, 0], c="C3", s=20, label=f"{name} mu")
        for t in idx:
            ax.plot([peak[t, 1], mu[t, 1]], [peak[t, 0], mu[t, 0]], "k-", lw=0.5, alpha=0.5)
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        ax.set_title(f"{name}: |mu - Lasso peak| = {dpk[idx].mean():.2f}px\n(shuffled-channel null "
                     f"{null:.2f}px, uniform {unif:.2f}px)", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{a.tier}_mu_vs_lasso.png"), dpi=130)
    # ALS spatial peak vs Lasso peak too
    for name in ("als_r1", "factgd_r1"):
        if name not in summ:
            continue
        z = np.load(os.path.join(od, f"{name}.npz"))
        s = np.load(os.path.join(od, f"{name}_summary.npz"))
        gsel = s["gsel"]
        pk = np.stack([peak_px(unit_weights(z, "fact", gsel[t], 0, d.shape)[..., t:t + 1])[0]
                       for t in range(len(gsel))])
        dd = np.linalg.norm(pk - peak, axis=1)
        nul = [np.linalg.norm(pk[idx] - peak[rng.permutation(idx)], axis=1) for _ in range(500)]
        res[name] = dict(dist_peak_lassopeak_px_mean_rel=float(dd[idx].mean()),
                         frac_same_position_rel=float((dd[idx] == 0).mean()),
                         null_shuffled_channels_px=float(np.mean([v.mean() for v in nul])),
                         null_frac_same_position=float(np.mean([(v == 0).mean() for v in nul])))
    with open(os.path.join(od, "mu_vs_lasso.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
