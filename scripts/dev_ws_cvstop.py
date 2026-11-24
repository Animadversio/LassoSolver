"""Compare torch_ws.fit_cv with and without CV early stopping (selection agreement + time) on a tier.
Usage: python scripts/dev_ws_cvstop.py tier [tol] [extra kw json]"""
import json, sys, time
import numpy as np
from lassobench import runner
from lassobench.api import STORE, alpha_grid
from lassobench.data import load_tier
from lassobench.solvers.torch_ws import TorchWS
import torch

tier = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
d = load_tier(tier)
X, Y = np.asarray(d.X), np.asarray(d.Y)
alphas = alpha_grid(X, Y, 50, 1e-2)
folds = runner.make_folds(X.shape[0])
rel = d.reliable_mask()
TorchWS(tol=tol).fit_path(X[:, :2000], Y[:, :4], alphas[:5, :4])
out = {}
for name, cs in [("full", "none"), ("stop10_5pct", (10, 0.05))]:
    s = TorchWS(tol=tol, cv_stop=cs, **kw)
    t0 = time.perf_counter(); r = s.fit_cv(X, Y, alphas, folds); dt = time.perf_counter() - t0
    pred = np.asarray(X_test := d.X_test) @ r["coef"].toarray()
    Yt = np.asarray(d.Y_test, np.float64)
    r2 = 1 - ((Yt - pred) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)
    fi = r["info"]["fold_infos"][0]
    out[name] = dict(time=dt, best=r["best_idx"], r2_rel=float(r2[rel].mean()), r2=float(r2.mean()),
                     alpha_stop_mean=fi.get("alpha_stop_mean"), timings={k: round(v, 1) for k, v in fi["timings"].items() if v > 0.5})
    torch.cuda.empty_cache()
a, b = out["full"], out["stop10_5pct"]
row = dict(tier=tier, tol=tol, kw=kw, gpu=torch.cuda.get_device_name(0), t_full=a["time"], t_stop=b["time"],
           best_agree=float(np.mean(a["best"] == b["best"])), n_disagree=int((a["best"] != b["best"]).sum()),
           r2_rel_full=a["r2_rel"], r2_rel_stop=b["r2_rel"], r2_full=a["r2"], r2_stop=b["r2"],
           alpha_stop_mean=b["alpha_stop_mean"], best_median=float(np.median(a["best"])), best_max=int(a["best"].max()),
           timings_full=a["timings"], timings_stop=b["timings"])
print(json.dumps(row), flush=True)
open(f"{STORE}/results/ws_dev/cvstop.jsonl", "a").write(json.dumps(row) + "\n")
