"""Markdown table from STORE/results/prox/sparsity_<tier>.json (written by dev_prox_sparsity.py).

Cells: mean test R2 over reliable channels. Delta columns: paired difference vs the relaxed exact Lasso
(lasso_ext path read at nnz ~ k, then ridge refit on its support), with the paired standard error over channels.
usage: python scripts/dev_prox_sparsity_table.py [unit-100k]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lassobench.api import STORE  # noqa: E402

tier = sys.argv[1] if len(sys.argv) > 1 else "unit-100k"
d = json.load(open(Path(STORE) / "results" / "prox" / f"sparsity_{tier}.json"))
pc = {k: np.asarray(v) for k, v in d["per_channel"].items()}
ks = d["ks"]
print(f"tier {tier}, {d['n_reliable']} reliable channels, fit {d['n_fit']} / val {d['n_val']} rows, {d['gpu']}; "
      f"ridge (all features) {d['ridge_test_r2_rel']:.3f}")
print("fit times s:", {k: round(v, 1) for k, v in d["fit_times_s"].items()})
rows = [("lasso_ext", "path"), ("lasso_ext", "path_refit"), ("lasso", "path"), ("lasso", "path_refit"),
        ("lasso_ext", "trunc"), ("lasso_ext", "refit"),
        ("prox_sgd", "trunc"), ("prox_sgd", "refit"), ("prox_adam", "trunc"), ("prox_adam", "refit"),
        ("hadamard", "trunc"), ("hadamard", "refit"), ("iht", "trunc"), ("iht", "refit"),
        ("iht_adam", "trunc"), ("iht_adam", "refit")]
print("\n| method | readout | " + " | ".join(f"k={k}" for k in ks) + " |")
print("|---|---|" + "---|" * len(ks))
for m, r in rows:
    cells = []
    for k in ks:
        v = pc.get(f"{m}|{k}|{r}")
        ref = pc.get(f"lasso_ext|{k}|path_refit")
        if v is None:
            cells.append("-")
            continue
        s = f"{v.mean():.3f}"
        if ref is not None and not (m == "lasso_ext" and r == "path_refit"):
            dd = v - ref
            s += f" ({dd.mean():+.3f}±{dd.std(ddof=1) / np.sqrt(len(dd)):.3f})"
        cells.append(s)
    print(f"| {m} | {r} | " + " | ".join(cells) + " |")
nnz = {m: d["table"][m] for m in ("lasso", "lasso_ext") if m in d["table"]}
print("\nLasso path nnz actually reached (median over reliable channels):",
      {m: {k: v[str(k)]["path_nnz"] if str(k) in v else v[k]["path_nnz"] for k in ks} for m, v in nnz.items()})
