"""Print a compact table of STORE/results/prox/<tag>.jsonl rows. usage: dev_prox_summary.py TAG [TAG...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lassobench.api import STORE  # noqa: E402

KEYS = ["q_mean", "q_max", "n_graph_captures", "time_wall_s", "time_engine_s", "total_iters", "slot_iters", "iters_max", "relgap_med", "relgap_max",
        "subopt_med", "subopt_max", "nnz_med", "r2_oracle", "r2_at_ref_alpha", "ref_r2_at_ref_alpha",
        "nnz_at_ref_alpha", "ref_nnz_at_ref_alpha", "n_unconverged", "peak_gpu_gb", "batch_capacity"]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


for tag in sys.argv[1:]:
    f = Path(STORE) / "results" / "prox" / f"{tag}.jsonl"
    for line in open(f):
        r = json.loads(line)
        head = f"{r['gpu'].replace('NVIDIA ', '')[:14]} n={r['n']} p={r['p']} T={r['T']} {r['solver']} {json.dumps(r['opts'])}"
        print(head)
        print("    " + "  ".join(f"{k}={fmt(r[k])}" for k in KEYS if r.get(k) is not None))
        if "phase_s" in r:
            print("    phases " + " ".join(f"{k}={v:.2f}" for k, v in r["phase_s"].items()))
