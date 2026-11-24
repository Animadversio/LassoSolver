"""Build dashboard/results.tsv = prior (Dec 2024) rows + Task CV rows from tables/results.jsonl.

For each (tier, solver, hardware, solver_args) the latest ok run is kept.
"""
import csv, json, pathlib
here = pathlib.Path(__file__).parent
cols = ["tier", "solver", "variant", "hardware", "device", "targets", "time_s", "test_r2", "test_r2_rel", "nnz", "source", "note"]
rows = []
for r in csv.DictReader(open(here / "results_prior.tsv"), delimiter="\t"):
    rows.append(dict(tier=r["tier"], solver=r["solver"], variant="", hardware=r["hardware"], device=r["device"], targets=r["targets"],
                     time_s=r["time_s"], test_r2=r["test_r2"], test_r2_rel="",
                     nnz=str(round(float(r["density"]) * 1000)) if r["density"] else "", source=r["source"], note=r["note"]))
latest = {}
for line in open(here.parent / "tables" / "results.jsonl"):
    r = json.loads(line)
    if r["task"] != "CV" or r["status"] != "ok":
        continue
    h, m, a = r["hardware"], r["metrics"], r["config"].get("solver_args") or {}
    shape = r.get("shape") or {}  # factorized round-2 rows carry no shape or timing
    if r["tier"] == "unit-800k" and shape.get("T") == 383:  # rows from before the tier was renamed
        r["tier"] = "unit-800k-383"
    gpu = h.get("gpu_model")
    hw = gpu.replace("NVIDIA ", "") if gpu else f"{h['n_threads']} CPU threads"
    args = ", ".join(f"{k}={v}" for k, v in sorted(a.items()))
    f = lambda v, n: "" if v is None else f"{v:.{n}g}"
    variant = ", ".join(f"{k}={v}" for k, v in sorted(a.items()) if k not in ("tol", "n_jobs", "lr", "finalize", "lr_schedule"))
    note = (args + "; " if args else "") + (f"50 alphas x {r['config']['n_folds']} folds, " if r['config'].get('n_folds') else "")
    note += f"p={shape['p']}" if shape else ""
    if "r2_splits_mean_reliable" in m:  # round 2: the row R2 is split 0; the split statistics go in the note
        variant = r["config"].get("label", "") + (", test columns clipped" if r["config"].get("clip_test") else "")
        note = (f"5 splits: {m['r2_splits_mean_reliable']:.3f} +- {m['r2_splits_sd_reliable']:.3f}"
                + (f", vs RidgeCV {m['diff_vs_ridgecv_splits_mean']:+.3f} (SE {m['diff_vs_ridgecv_splits_se']:.3f})"
                   if m.get("diff_vs_ridgecv_splits_mean") is not None else "")
                + (f"; nonzeros on reliable channels {m['nnz_mean_reliable']:.0f}" if m.get("nnz_mean_reliable") else ""))
    latest[(r["tier"], shape.get("T"), r["solver"], hw, args, r["config"].get("clip_test"))] = dict(
        tier=r["tier"], solver=r["solver"], variant=variant, hardware=hw, device="gpu" if gpu else "cpu", targets=str(shape.get("T", 64)),
        time_s=f(r.get("time_total_s"), 3), test_r2=f(m.get("r2_test_mean"), 3), test_r2_rel=f(m.get("r2_test_mean_reliable"), 3),
        nnz=f(m.get("nnz_mean"), 3), source="harness " + r["timestamp"][:10],
        note=note)
rows += latest.values()
for r in rows:  # disambiguate repeated (tier, solver, hardware, targets) for the dashboard's row key
    pass
with open(here / "results.tsv", "w") as fo:
    w = csv.DictWriter(fo, cols, delimiter="\t"); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
