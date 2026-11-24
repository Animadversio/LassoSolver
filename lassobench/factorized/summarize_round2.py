"""Tables, rows and figures for round 2 (reads STORE/results/factorized/<tier>/round2/*.npz from `round2.py`).

  python -m lassobench.factorized.summarize_round2 --tier unit-100k [--write-rows]

Prints markdown tables (splits, paired differences vs RidgeCV, stability by source, ablations), writes
round2/summary.json, figures/factorized/<tier>_round2_{splits,stability}.png and, with --write-rows, one row per
model in tables/results.jsonl (solver fact_r2_<model>, task CV, tag factorized-readout-round2; the row's
r2_test_mean_reliable is split 0 = the tier's own split, the split statistics are extra metric keys).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import uuid

import numpy as np

from lassobench.factorized.data import OUT_DIR

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(REPO, "figures", "factorized")

SPLIT_MODELS = [  # (name, label, how to get the (n_te, T) prediction from a split npz)
    ("ridgecv", "RidgeCV (single, 100%)", lambda z: z["ridgecv_pred"]),
    ("ridge_bag5", "Ridge, 5 bagged 80% fits", lambda z: z["ridgebag_mem_pred"].mean(0)),
    ("lasso_cv", "Lasso CV torch_ws (single, 100%)", lambda z: z["lassocv_pred"]),
    ("lasso_mem5", "Lasso 5-member ens.", lambda z: z["lassomem_pred"].mean(0)),
    ("factgd_r1_single", "factgd r1 single (100% refit)", lambda z: z["r1_single_pred"]),
    ("factgd_r1_ens5", "factgd r1 5-member ens.", lambda z: z["r1_mem_pred"].mean(0)),
    ("factgd_r2_single", "factgd r2 single (100% refit)", lambda z: z["r2_single_pred"]),
    ("factgd_r2_ens5", "factgd r2 5-member ens.", lambda z: z["r2_mem_pred"].mean(0)),
    ("nuclear_single", "nuclear-norm single (100% refit)", None),
    ("nuclear_ens5", "nuclear-norm 5-member ens.", None),
]
MEMBER_MODELS = [  # single 80% members, mean over members of the per-member R2
    ("ridge_member80", "Ridge one 80% fit", "ridgebag_mem_pred"),
    ("lasso_member80", "Lasso one 80% member", "lassomem_pred"),
    ("factgd_r1_member80", "factgd r1 one 80% member", "r1_mem_pred"),
    ("factgd_r2_member80", "factgd r2 one 80% member", "r2_mem_pred"),
    ("nuclear_member80", "nuclear-norm one 80% member", None),
]


def r2(Yt, P):
    Yt = np.asarray(Yt, np.float64)
    return 1 - ((Yt - P) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)


def pcorr(P1, P2):
    a, b = P1 - P1.mean(0), P2 - P2.mean(0)
    return (a * b).sum(0) / np.maximum(np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0), 1e-12)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def subspace_sim(A, B):
    """A, B (r, d): mean squared cosine of the principal angles between their row spaces (|cos| for r = 1)."""
    keepA, keepB = np.linalg.norm(A, axis=1) > 0, np.linalg.norm(B, axis=1) > 0
    A, B = A[keepA], B[keepB]
    if len(A) == 0 or len(B) == 0:
        return np.nan
    if len(A) == 1 and len(B) == 1:
        return abs(cos(A[0], B[0]))
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float((s ** 2).sum() / max(len(A), len(B)))


def top_factors(Wt, shape, r=1):
    """Wt (p,) unit-basis weights -> rank-r SVD factors of the C x HW matrix: (spatial (r, HW), feature (r, C))."""
    C, H, W = shape
    M = Wt.reshape(C, H * W)
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    return Vh[:r], U[:, :r].T


def recon(S, F, shape):
    """S (r, HW), F (r, C) -> (C*HW,) unit weights in (c, h, w) order."""
    return np.einsum("rs,rc->cs", S, F).ravel()


def pair_stats(runs, chans, shape, kind):
    """runs: list of dicts with 'pred' (n_te, T) and either S/F (fact) or W (p, T). Mean over pairs, per channel."""
    out = {k: [] for k in ("mask", "feature", "wcos", "jaccard", "predcorr")}
    for t in chans:
        acc = {k: [] for k in out}
        for i, j in itertools.combinations(range(len(runs)), 2):
            a, b = runs[i], runs[j]
            if kind == "fact":
                Sa, Fa, Sb, Fb = a["S"][t], a["F"][t], b["S"][t], b["F"][t]
                Wa, Wb = recon(Sa, Fa, shape), recon(Sb, Fb, shape)
            else:
                Wa, Wb = a["W"][:, t], b["W"][:, t]
                r = a.get("r", 1)
                Sa, Fa = top_factors(Wa, shape, r)
                Sb, Fb = top_factors(Wb, shape, r)
            acc["mask"].append(subspace_sim(Sa, Sb))
            acc["feature"].append(subspace_sim(Fa, Fb))
            acc["wcos"].append(cos(Wa, Wb))
            sa, sb = Wa != 0, Wb != 0
            u = (sa | sb).sum()
            dense = sa.mean() > 0.9 and sb.mean() > 0.9                  # Ridge / nuclear: dense, no support
            acc["jaccard"].append((sa & sb).sum() / u if u and not dense else np.nan)
            acc["predcorr"].append(pcorr(a["pred"][:, t:t + 1], b["pred"][:, t:t + 1])[0])
        for k in out:
            out[k].append(np.nanmean(acc[k]))
    return {k: np.asarray(v) for k, v in out.items()}


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def split_tables(od, tier):
    files = sorted(f for f in os.listdir(od) if f.startswith("split") and f.endswith(".npz"))
    ks = sorted(int(f[5:-4]) for f in files)
    per = {}       # name -> list over splits of per-channel R2 on reliable channels
    nnz = {}
    meta0 = None
    for k in ks:
        z = np.load(os.path.join(od, f"split{k}.npz"))
        if k == 0:
            meta0 = json.loads(str(z["meta"]))
        rel = z["reliable"]
        Yte = z["Yte"]
        for name, _, f in SPLIT_MODELS:
            if f is not None:
                per.setdefault(name, []).append(r2(Yte, f(z))[rel])
        for name, _, key in MEMBER_MODELS:
            if key is not None:
                per.setdefault(name, []).append(np.mean([r2(Yte, p)[rel] for p in z[key]], 0))
        nf = os.path.join(od, f"nuclear_split{k}.npz")
        if os.path.exists(nf):
            zn = np.load(nf)
            per.setdefault("nuclear_single", []).append(r2(Yte, zn["nuclear_single_pred"])[rel])
            per.setdefault("nuclear_ens5", []).append(r2(Yte, zn["nuclear_mem_pred"].mean(0))[rel])
            per.setdefault("nuclear_member80", []).append(np.mean([r2(Yte, p)[rel] for p in zn["nuclear_mem_pred"]], 0))
            nnz.setdefault("nuclear_rank", []).append(zn["nuclear_single_rank"][rel].mean())
        nnz.setdefault("lasso_cv", []).append(z["lassocv_nnz"][rel].mean())
        nnz.setdefault("lasso_mem5", []).append(z["lassomem_nnz"][:, rel].mean())
        nnz.setdefault("lasso_member80", []).append(z["lassomem_nnz"][:, rel].mean())
        for r in (1, 2):
            nnz.setdefault(f"factgd_r{r}_single", []).append(z[f"r{r}_single_eff_units"][rel].mean())
            nnz.setdefault(f"factgd_r{r}_ens5", []).append(z[f"r{r}_mem_eff_units"][:, rel].mean())
            nnz.setdefault(f"factgd_r{r}_member80", []).append(z[f"r{r}_mem_eff_units"][:, rel].mean())
    ref = per["ridgecv"]
    rows = {}
    for name in [m[0] for m in SPLIT_MODELS] + [m[0] for m in MEMBER_MODELS]:
        if name not in per or len(per[name]) != len(ks):
            continue
        v = per[name]
        means = np.array([x.mean() for x in v])
        dif = [x - y for x, y in zip(v, ref)]
        dm = np.array([x.mean() for x in dif])
        se_ch = np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in dif])
        rows[name] = dict(split_means=means.tolist(), mean=float(means.mean()),
                          sd=float(means.std(ddof=1)) if len(means) > 1 else None,
                          diff_vs_ridge_split0=float(dm[0]), se_ch_split0=float(se_ch[0]),
                          diff_vs_ridge_mean=float(dm.mean()),
                          diff_sd_splits=float(dm.std(ddof=1)) if len(dm) > 1 else None,
                          se_splits=float(dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 1 else None,
                          se_ch_mean=float(se_ch.mean()), n_splits=len(ks), n_better_splits=int((dm > 0).sum()),
                          nnz_mean=float(np.mean(nnz[name])) if name in nnz else None)
        if name.startswith("nuclear") and "nuclear_rank" in nnz:
            rows[name]["rank_single_mean"] = float(np.mean(nnz["nuclear_rank"]))
    # ensemble vs ensemble: factgd ens5 - ridge bag5
    for r in (1, 2):
        dif = [x - y for x, y in zip(per[f"factgd_r{r}_ens5"], per["ridge_bag5"])]
        dm = np.array([x.mean() for x in dif])
        rows[f"factgd_r{r}_ens5"]["diff_vs_ridgebag_mean"] = float(dm.mean())
        rows[f"factgd_r{r}_ens5"]["diff_vs_ridgebag_se_splits"] = float(dm.std(ddof=1) / np.sqrt(len(dm))) \
            if len(dm) > 1 else None
    return ks, rows, per, meta0


def stability_tables(od, shape):
    z = np.load(os.path.join(od, "stab.npz"))
    rel = np.flatnonzero(z["reliable"])
    Yte = z["Yte"]
    names = [str(s) for s in z["subset_names"]]
    sub = [i for i, s in enumerate(names) if s.startswith("sub")]
    half = [i for i, s in enumerate(names) if s.startswith("half")]
    res = {}

    def summarize(runs, kind, label):
        st = pair_stats(runs, rel, shape, kind)
        r2s = [r2(Yte, r["pred"])[rel].mean() for r in runs]
        res[label] = {k: float(np.nanmean(v)) for k, v in st.items()}
        res[label]["r2_mean"], res[label]["r2_sd"] = float(np.mean(r2s)), float(np.std(r2s, ddof=1))
        res[label]["n_runs"] = len(runs)
        res[label]["per_channel"] = {k: v.tolist() for k, v in st.items()}

    for r in (1, 2, 3):
        if f"r{r}_seed_S" not in z.files:
            continue
        for src in ("seed", "data"):
            S, F, P = z[f"r{r}_{src}_S"], z[f"r{r}_{src}_F"], z[f"r{r}_{src}_pred"]
            idxs = {"seed": [("seed", list(range(len(S))))],
                    "data": [("data80", sub), ("half", half)]}[src]
            for lab, ii in idxs:
                runs = [dict(S=S[i], F=F[i], pred=P[i]) for i in ii]
                summarize(runs, "fact", f"factgd_r{r}|{lab}")
    for meth in ("ridge", "lasso"):
        W, P = z[f"{meth}_data_W"], z[f"{meth}_data_pred"]
        for lab, ii in (("data80", sub), ("half", half)):
            for rr in (1, 2):
                runs = [dict(W=W[i], pred=P[i], r=rr) for i in ii]
                summarize(runs, "lin", f"{meth}(svd r{rr})|{lab}")
    nf = os.path.join(od, "nuclear_split0.npz")
    if os.path.exists(nf):
        zn = np.load(nf)
        nn = [str(x) for x in zn["subset_names"]]
        for lab, ii in (("data80", [i for i, x in enumerate(nn) if x.startswith("sub")]),
                        ("half", [i for i, x in enumerate(nn) if x.startswith("half")])):
            for rr in (1, 2):
                runs = [dict(W=zn["nuclear_data_W"][i], pred=zn["nuclear_data_pred"][i], r=rr) for i in ii]
                summarize(runs, "lin", f"nuclear(svd r{rr})|{lab}")
    # cross-method: factgd r1 (data runs, sub0) factors vs Ridge(full) rank-1 SVD factors
    cross = {"mask": [], "feature": []}
    Wr = z["ridge_full_W"]
    for t in rel:
        s_r, f_r = top_factors(Wr[:, t], shape, 1)
        cross["mask"].append(subspace_sim(z["r1_seed_S"][0, t], s_r))
        cross["feature"].append(subspace_sim(z["r1_seed_F"][0, t], f_r))
    res["cross_factgd_r1_vs_ridge_svd1"] = {k: float(np.nanmean(v)) for k, v in cross.items()}
    return res


def ablation_table(od):
    z = np.load(os.path.join(od, "ablate.npz"))
    rel = z["reliable"]
    Yte = z["Yte"]
    ref = r2(Yte, z["ridge_pred"])[rel]
    out = {}

    def add(name, P, eff=None, extra=None):
        v = r2(Yte, P)[rel]
        d = v - ref
        out[name] = dict(r2=float(v.mean()), diff_vs_ridge=float(d.mean()), se_ch=float(d.std(ddof=1) / np.sqrt(len(d))),
                         eff_units=None if eff is None else float(np.mean(eff)), **(extra or {}))

    add("ridgecv", z["ridge_pred"])
    names = sorted({k[: -len("_single_pred")] for k in z.files if k.endswith("_single_pred")})
    for n in names:
        if n == "nuclear":
            add("nuclear_single", z["nuclear_single_pred"], extra=dict(rank=float(z["nuclear_single_rank"][rel].mean())))
            add("nuclear_ens5", z["nuclear_mem_pred"].mean(0))
            continue
        add(f"{n}_single", z[f"{n}_single_pred"], z[f"{n}_single_eff_units"][rel])
        add(f"{n}_ens5", z[f"{n}_mem_pred"].mean(0), z[f"{n}_mem_eff_units"][:, rel])
    for r in (1, 2, 3):
        add(f"ridge_tsvd{r}", z[f"ridge_tsvd{r}_pred"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--dir", default="round2")
    ap.add_argument("--write-rows", action="store_true")
    ap.add_argument("--tag", default="factorized-readout-round2")
    a = ap.parse_args(argv)
    od = os.path.join(OUT_DIR, a.tier, a.dir)
    T0 = time.perf_counter()
    summ = {}
    ks, rows, per, meta0 = split_tables(od, a.tier)
    summ["splits"] = rows
    shape = None
    from lassobench.factorized.data import LAYER_MAPS
    from lassobench.data import load_tier
    shape = LAYER_MAPS[load_tier(a.tier).meta["layer"]][0]
    print(f"\n## {a.tier}: test R2 on reliable channels over {len(ks)} splits (splits {ks})\n")
    print("| model | split 0 | mean ± SD over splits | diff vs RidgeCV, split 0 (SE ch.) | "
          "diff vs RidgeCV, mean over splits (SE splits) | splits better | nonzero units |")
    print("|---|---|---|---|---|---|---|")
    for name, lab, *_ in SPLIT_MODELS + MEMBER_MODELS:
        if name not in rows:
            continue
        r = rows[name]
        nz = ("dense" if name.startswith("ridge") else "n/a") if r["nnz_mean"] is None else f"{r['nnz_mean']:.0f}"
        print(f"| {lab} | {r['split_means'][0]:.3f} | {r['mean']:.3f} ± {fmt(r['sd'])} | "
              f"{r['diff_vs_ridge_split0']:+.3f} ({r['se_ch_split0']:.3f}) | {r['diff_vs_ridge_mean']:+.3f} "
              f"({fmt(r['se_splits'])}) | {r['n_better_splits']}/{r['n_splits']} | {nz} |")
    for r_ in (1, 2):
        e = rows[f"factgd_r{r_}_ens5"]
        print(f"\nfactgd r{r_} ens5 - Ridge bag5: {e['diff_vs_ridgebag_mean']:+.3f} (SE splits "
              f"{fmt(e['diff_vs_ridgebag_se_splits'])})")
    print(f"[{time.perf_counter() - T0:.0f}s] splits done", flush=True)
    if os.path.exists(os.path.join(od, "stab.npz")):
        st = stability_tables(od, shape)
        summ["stability"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_channel"} for k, v in st.items()}
        summ["stability_per_channel"] = {k: v.get("per_channel") for k, v in st.items()}
        print(f"\n## {a.tier}: stability on split 0 (mean over reliable channels of the mean over run pairs)\n")
        print("| model \\| source | runs | test R2 mean ± SD | spatial mask sim. | feature vector sim. | "
              "weight cosine | support Jaccard | test-prediction corr. |")
        print("|---|---|---|---|---|---|---|---|")
        for k, v in st.items():
            if k.startswith("cross"):
                continue
            print(f"| {k} | {v['n_runs']} | {v['r2_mean']:.3f} ± {v['r2_sd']:.3f} | {fmt(v['mask'])} | "
                  f"{fmt(v['feature'])} | {fmt(v['wcos'])} | {fmt(v['jaccard'])} | {fmt(v['predcorr'])} |")
        print(f"\ncross-method (factgd r1 seed-1 refit vs Ridge rank-1 SVD factors): "
              f"{json.dumps(st['cross_factgd_r1_vs_ridge_svd1'])}")
    print(f"[{time.perf_counter() - T0:.0f}s] stability done", flush=True)
    if os.path.exists(os.path.join(od, "ablate.npz")):
        ab = ablation_table(od)
        summ["ablate"] = ab
        print(f"\n## {a.tier}: ablations on split 0\n")
        print("| variant | test R2 rel | diff vs RidgeCV (SE ch.) | eff. units | rank |")
        print("|---|---|---|---|---|")
        for k, v in ab.items():
            print(f"| {k} | {v['r2']:.3f} | {v['diff_vs_ridge']:+.3f} ({v['se_ch']:.3f}) | "
                  f"{fmt(v['eff_units'], 0)} | {fmt(v.get('rank'), 2)} |")
    tp = os.path.join(OUT_DIR, a.tier, a.dir, "timing.npz")
    if os.path.exists(tp):
        summ["timing"] = json.loads(str(np.load(tp)["meta"]))
    summ["meta_split0"] = meta0
    with open(os.path.join(od, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    figures(a.tier, ks, per, summ)
    if a.write_rows:
        write_rows(a.tier, rows, summ, meta0, tag=a.tag, raw_dir=od)


def figures(tier, ks, per, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [m[0] for m in SPLIT_MODELS if m[0] in per and len(per[m[0]]) == len(ks)]
    for i, n in enumerate(names):
        y = [v.mean() for v in per[n]]
        ax.plot(np.full(len(y), i) + np.linspace(-0.2, 0.2, len(y)), y, "o", ms=5, label=None)
        ax.plot([i - 0.3, i + 0.3], [np.mean(y)] * 2, "k-", lw=2)
    for k in range(len(ks)):
        ax.plot(range(len(names)), [per[n][k].mean() for n in names], "-", color="gray", lw=0.5, alpha=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([m[1] for m in SPLIT_MODELS if m[0] in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test R2, mean over reliable channels")
    ax.set_title(f"{tier}: {len(ks)} train/test splits (grey lines join the same split)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{tier}_round2_splits.png"), dpi=130)
    st = summ.get("stability")
    if not st:
        return
    labels = [k for k in st if not k.startswith("cross")]
    mets = ["mask", "feature", "wcos", "jaccard", "predcorr"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    w = 0.8 / len(mets)
    for j, m in enumerate(mets):
        ax.bar(np.arange(len(labels)) + j * w, [st[k][m] if st[k][m] is not None else np.nan for k in labels],
               w, label=m)
    ax.set_xticks(np.arange(len(labels)) + 0.4 - w / 2)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, ncol=5)
    ax.set_title(f"{tier}: stability by source (split 0, reliable channels). mask/feature = |cos| (rank 1) or "
                 f"subspace similarity; Ridge/Lasso factors = SVD of C x HW weights", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{tier}_round2_stability.png"), dpi=130)


def write_rows(tier, rows, summ, meta0, tag="factorized-readout-round2", raw_dir=None):
    from lassobench import runner
    hw = (meta0 or {}).get("hardware")
    for name, lab, *_ in SPLIT_MODELS + MEMBER_MODELS:
        if name not in rows:
            continue
        r = rows[name]
        m = dict(r2_test_mean_reliable=r["split_means"][0], r2_splits_mean_reliable=r["mean"],
                 r2_splits_sd_reliable=r["sd"], diff_vs_ridgecv_split0=r["diff_vs_ridge_split0"],
                 diff_vs_ridgecv_split0_se_channels=r["se_ch_split0"],
                 diff_vs_ridgecv_splits_mean=r["diff_vs_ridge_mean"],
                 diff_vs_ridgecv_splits_se=r["se_splits"], n_splits=r["n_splits"],
                 nnz_mean_reliable=r["nnz_mean"])
        for k in ("diff_vs_ridgebag_mean", "diff_vs_ridgebag_se_splits"):
            if k in r:
                m[k] = r[k]
        if name.startswith("factgd") and summ.get("stability"):
            rk = name.split("_")[1]
            for src in ("seed", "data80", "half"):
                s = summ["stability"].get(f"factgd_{rk}|{src}")
                if s:
                    for kk in ("mask", "feature", "wcos", "jaccard", "predcorr"):
                        m[f"stab_{src}_{kk}"] = s[kk]
        row = dict(run_id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), task="CV", solver=f"fact_r2_{name}", tier=tier,
                   tag=tag, config=dict(label=lab, protocol="round2 (see round2.py)",
                                        clip_test=bool((meta0 or {}).get("clip_test", False))),
                   hardware=hw, git_hash=(meta0 or {}).get("git_hash"), git_dirty=(meta0 or {}).get("git_dirty"),
                   status="ok", metrics=m, raw=raw_dir or os.path.join(OUT_DIR, tier, "round2"))
        runner.append_row(row)
    print(f"wrote {sum(n in rows for n, *_ in SPLIT_MODELS + MEMBER_MODELS)} rows")


if __name__ == "__main__":
    main()
