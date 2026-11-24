"""celer yardstick (harness cpu_celer, process-parallel) on a dataset saved by dev_ws_scale.py --save_data.
Times the first m targets with n_jobs processes; reports s/target (wall * n_jobs / m = per-target CPU-time)
and the projected wall time for all T targets on n_jobs cores.
Usage: python scripts/dev_ws_celer_saved.py DIR tol m n_jobs"""
import json, os, sys, time, socket
import numpy as np
from lassobench.api import STORE
from lassobench.solvers.cpu_celer import Celer
d, tol, m, nj = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
X = np.load(os.path.join(d, "X.npy")); Y = np.load(os.path.join(d, "Y.npy"))
n, p = X.shape; T = Y.shape[1]
amax = np.abs(X.T @ Y).max(0).astype(np.float64) / n
alphas = amax[None, :] * np.logspace(0, -2, 50)[:, None]
s = Celer(tol=tol, n_jobs=nj)
t0 = time.perf_counter(); s.fit_path(X, Y[:, :m], alphas[:, :m]); wall = time.perf_counter() - t0
cpu = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ")
row = dict(solver="cpu_celer", data=os.path.basename(d), n=n, p=p, T=T, tol=tol, m_targets=m, n_jobs=nj, wall_s=wall,
           s_per_target_cpu=wall * nj / m, proj_wall_T_on_njobs=wall * T / m, proj_single_thread_T=wall * nj / m * T,
           cpu=cpu, host=socket.gethostname())
print(json.dumps(row), flush=True)
open(f"{STORE}/results/ws_dev/celer_yard.jsonl", "a").write(json.dumps(row) + "\n")
