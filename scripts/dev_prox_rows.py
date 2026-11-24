"""Print gpu-prox harness rows from tables/results.jsonl. usage: dev_prox_rows.py [tier-prefix]"""
import json
import sys
from pathlib import Path

pre = sys.argv[1] if len(sys.argv) > 1 else ""
KM = ["gap_rel_median", "gap_rel_max", "frac_problems_loose", "nnz_median", "subopt_rel_median", "subopt_rel_max",
      "r2_test_mean_reliable", "r2_test_mean", "nnz_mean", "jaccard_ref_mean", "best_idx_median"]
KI = ["total_iters", "n_unconverged", "q_mean", "q_max", "n_screen_rebuilds", "time_engine_s"]
for l in open(Path(__file__).resolve().parents[1] / "tables" / "results.jsonl"):
    r = json.loads(l)
    if r.get("tag") != "gpu-prox" or not r["tier"].startswith(pre):
        continue
    m = r.get("metrics") or {}
    si = r.get("solver_info") or {}
    t = r.get("time_total_s")
    print(f"{r['solver']:20s} {r['task']:2s} {r['tier']:10s} {(r['hardware'].get('gpu_model') or '').replace('NVIDIA ', ''):14s} "
          f"{r['status']:7s} t={t if t is None else round(t, 1)} git={r['git_hash'][:7]} {r['config']['solver_args']}")
    print("      " + " ".join(f"{k}={m[k]:.4g}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}" for k in KM if k in m)
          + " | " + " ".join(f"{k}={si[k]:.4g}" if isinstance(si.get(k), float) else f"{k}={si.get(k)}" for k in KI if k in si))
