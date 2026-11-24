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
    if r["tier"] == "unit-800k" and r["shape"]["T"] == 383:  # rows from before the tier was renamed
        r["tier"] = "unit-800k-383"
    gpu = h.get("gpu_model")
    hw = gpu.replace("NVIDIA ", "") if gpu else f"{h['n_threads']} CPU threads"
    args = ", ".join(f"{k}={v}" for k, v in sorted(a.items()))
    f = lambda v, n: "" if v is None else f"{v:.{n}g}"
    variant = ", ".join(f"{k}={v}" for k, v in sorted(a.items()) if k not in ("tol", "n_jobs", "lr", "finalize", "lr_schedule"))
    latest[(r["tier"], r["shape"]["T"], r["solver"], hw, args)] = dict(
        tier=r["tier"], solver=r["solver"], variant=variant, hardware=hw, device="gpu" if gpu else "cpu", targets=str(r["shape"]["T"]),
        time_s=f(r["time_total_s"], 3), test_r2=f(m.get("r2_test_mean"), 3), test_r2_rel=f(m.get("r2_test_mean_reliable"), 3),
        nnz=f(m.get("nnz_mean"), 3), source="harness " + r["timestamp"][:10],
        note=(args + "; " if args else "") + (f"50 alphas x {r['config']['n_folds']} folds, " if r['config'].get('n_folds') else "") + f"p={r['shape']['p']}")
rows += latest.values()
for r in rows:  # disambiguate repeated (tier, solver, hardware, targets) for the dashboard's row key
    pass
with open(here / "results.tsv", "w") as fo:
    w = csv.DictWriter(fo, cols, delimiter="\t"); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
