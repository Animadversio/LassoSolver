"""Benchmark runner (owned by bench-harness): solver discovery, Task P / Task CV, timing, result rows.

CV protocol extension (proposed to the coordinator): a solver may define

    fit_cv(X, Y, alphas, folds) -> dict(coef=(p, T) sparse or dense refit at chosen alphas,
                                        best_idx=(T,) int index into alphas rows,
                                        cv_mse=(A, T) mean held-out MSE, info=dict)

where folds is the harness list of (train_idx, val_idx). Otherwise `default_fit_cv` is used, which calls
fit_path K+1 times (K folds + one refit on all training rows, truncated to the largest chosen alpha index).
"""
from __future__ import annotations

import importlib
import json
import os
import pkgutil
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid

import numpy as np
import scipy.sparse as sp

from lassobench import api, evaluate as ev
from lassobench.data import Tier, load_tier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Result log: one JSON row per run. Set LASSOBENCH_RESULTS to write your own file (avoids merge conflicts in the
# shared tables/results.jsonl); `python -m lassobench.report <files...>` can combine several.
RESULTS = os.environ.get("LASSOBENCH_RESULTS", os.path.join(REPO, "tables", "results.jsonl"))
RAW_DIR = os.path.join(api.STORE, "results", "raw")   # always under the writable STORE, never DATA_ROOT
N_FOLDS, FOLD_SEED = 5, 0
N_ALPHAS, ALPHA_EPS = 50, 1e-2


# ----------------------------------------------------------------------------- discovery

def discover_solvers(verbose=False):
    """Import every module in lassobench.solvers and merge their module-level SOLVERS dicts.

    A module that fails to import is skipped (recorded in the returned `errors`), so one broken
    work-in-progress file does not take the harness down.
    """
    import lassobench.solvers as pkg
    solvers, errors = {}, {}
    for m in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"lassobench.solvers.{m.name}")
        except Exception as e:  # noqa: BLE001
            errors[m.name] = f"{type(e).__name__}: {e}"
            continue
        for k, v in getattr(mod, "SOLVERS", {}).items():
            solvers[k] = v
    if verbose and errors:
        print("solver modules that failed to import:", errors, file=sys.stderr)
    return solvers, errors


def get_solver(name, **kw):
    solvers, errors = discover_solvers()
    if name not in solvers:
        raise KeyError(f"unknown solver {name!r}. registered: {sorted(solvers)}; import errors: {errors}")
    return solvers[name](**kw)


# ----------------------------------------------------------------------------- environment

def git_info():
    def g(*a):
        try:
            return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    dirty = [l[3:] for l in g("status", "--porcelain").splitlines() if l[3:].startswith(("lassobench/", "scripts/"))]
    return dict(git_hash=g("rev-parse", "HEAD"), git_dirty=dirty)


def hardware_info():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    hw = dict(hostname=socket.gethostname(), cpu_model=cpu, n_threads=len(os.sched_getaffinity(0)),
              slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
              gpu_model=None, n_gpus=0)
    try:
        import pynvml
        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        cnt = pynvml.nvmlDeviceGetCount()
        idx = [int(i) for i in vis.split(",") if i.strip().isdigit()] if vis else list(range(cnt))
        names = [pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idx if i < cnt]
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        if names:
            hw.update(gpu_model=names[0], n_gpus=len(names))
    except Exception:  # noqa: BLE001
        pass
    return hw


class MemSampler:
    """Samples host RSS (process + children) and this process's GPU memory (NVML) in a thread."""

    def __init__(self, interval=0.2):
        import psutil
        self.proc = psutil.Process()
        self.interval = interval
        self.peak_rss = 0
        self.peak_gpu = 0
        self._pss_ok, self._use_pss = True, False
        self._stop = threading.Event()
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:  # noqa: BLE001
            self._nvml = None

    def _mem(self, p):
        # PSS (shared pages split across processes) when there are worker processes, else RSS.
        # Summing RSS over workers that share a memmap of X overcounts it once per worker.
        if self._use_pss:
            try:
                return p.memory_full_info().pss
            except Exception:  # noqa: BLE001
                self._use_pss = False
        return p.memory_info().rss

    def _rss(self):
        tot = 0
        try:
            kids = self.proc.children(recursive=True)
            self._use_pss = bool(kids) and self._pss_ok
            tot = self._mem(self.proc)
            for c in kids:
                try:
                    tot += self._mem(c)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return tot

    def _gpu(self):
        if self._nvml is None:
            return 0
        pid, tot = os.getpid(), 0
        for h in self._handles:
            try:
                for pr in self._nvml.nvmlDeviceGetComputeRunningProcesses(h):
                    if pr.pid == pid and pr.usedGpuMemory:
                        tot += pr.usedGpuMemory
            except Exception:  # noqa: BLE001
                pass
        return tot

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self._rss())
            self.peak_gpu = max(self.peak_gpu, self._gpu())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.rss_before = self._rss()
        self.peak_rss = self.rss_before
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak_rss = max(self.peak_rss, self._rss())
        self.peak_gpu = max(self.peak_gpu, self._gpu())
        torch = sys.modules.get("torch")
        self.torch_peak = None
        if torch is not None and torch.cuda.is_available():
            self.torch_peak = torch.cuda.max_memory_allocated()

    def summary(self):
        return dict(ram_before_gb=self.rss_before / 2**30, peak_ram_gb=self.peak_rss / 2**30, ram_metric="rss+pss(workers)",
                    peak_gpu_mem_gb=(max(self.peak_gpu, self.torch_peak or 0) / 2**30) if
                    (self.peak_gpu or self.torch_peak) else None)


class Timeout(Exception):
    pass


def _alarm(sig, frame):
    raise Timeout()


def _sync(solver):
    if getattr(solver, "device", "cpu") == "cuda":
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()


# ----------------------------------------------------------------------------- tasks

def make_folds(n, k=N_FOLDS, seed=FOLD_SEED):
    from sklearn.model_selection import KFold
    return [(np.asarray(tr), np.asarray(va)) for tr, va in KFold(k, shuffle=True, random_state=seed).split(np.arange(n))]


def check_path_result(res, p, T, A):
    assert isinstance(res, api.PathResult), type(res)
    assert len(res.coefs) == A, (len(res.coefs), A)
    for c in res.coefs:
        assert c.shape == (p, T), c.shape


def _timed_fit_path(solver, X, Y, alphas):
    t0 = time.perf_counter()
    res = solver.fit_path(X, Y, alphas)
    _sync(solver)
    return res, time.perf_counter() - t0


def default_fit_cv(solver, X, Y, alphas, folds, keep_paths=False):
    A, T = alphas.shape
    mse = np.zeros((len(folds), A, T))
    info = dict(fold_times_s=[], fold_infos=[])
    fold_paths = []
    for k, (tr, va) in enumerate(folds):
        Xtr, Ytr = np.ascontiguousarray(X[tr]), np.ascontiguousarray(Y[tr])
        res, dt = _timed_fit_path(solver, Xtr, Ytr, alphas)
        check_path_result(res, X.shape[1], T, A)
        Xva, Yva = np.asarray(X[va]), np.asarray(Y[va], np.float64)
        for a in range(A):
            mse[k, a] = ((Yva - ev.predict(Xva, res.coefs[a])) ** 2).mean(0)
        info["fold_times_s"].append(dt)
        info["fold_infos"].append(_small(res.info))
        if keep_paths:
            fold_paths.append([sp.csc_matrix(c) for c in res.coefs])
        del Xtr, Ytr, res
    cv_mse = mse.mean(0)
    best = np.argmin(cv_mse, axis=0)                  # first minimum = largest alpha among ties
    amax = A - 1 if keep_paths else int(best.max())   # analysis needs the full refit path
    res, dt = _timed_fit_path(solver, np.asarray(X), np.asarray(Y), np.ascontiguousarray(alphas[: amax + 1]))
    check_path_result(res, X.shape[1], T, amax + 1)
    cols = [sp.csc_matrix(res.coefs[best[t]])[:, t] for t in range(T)]
    info.update(refit_time_s=dt, refit_info=_small(res.info), refit_full_path=bool(keep_paths))
    out = dict(coef=sp.hstack(cols).tocsc(), best_idx=best, cv_mse=cv_mse, info=info)
    if keep_paths:
        out.update(fold_paths=fold_paths, path_coefs=[sp.csc_matrix(c) for c in res.coefs])
    return out


def half_split(n, seed=1):
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[: n // 2]), np.sort(perm[n // 2:])]


def run_analysis(solver, tier, X, Y, alphas, folds, result):
    """Untimed round-2 analysis (spec section 9). Returns (summary, per-channel arrays, seconds)."""
    from lassobench.solvers.cpu_ridgecv import RidgeCVBaseline
    t0 = time.perf_counter()
    rel = tier.reliable_mask()
    rr = RidgeCVBaseline().fit_cv(X, Y, None, None)
    ridge = dict(coef=rr["coef"], r2=ev.r2_per_channel(tier.Y_test, ev.predict(tier.X_test, rr["coef"])),
                 fold_coefs=[RidgeCVBaseline().fit_cv(np.asarray(X[tr]), np.asarray(Y[tr]), None, None)["coef"]
                             for tr, va in folds])
    halves = half_split(X.shape[0])
    has_path = type(solver).fit_path is not api.Solver.fit_path
    if not has_path:  # dense / own-grid methods: top-k frontier of their own coefficients + ridge reference
        r2 = ev.r2_per_channel(tier.Y_test, ev.predict(tier.X_test, result["coef"]))
        S = dict(ridge_r2_reliable=ev._rel_stats(ridge["r2"], rel)["mean"],
                 own_r2_reliable=ev._rel_stats(r2, rel)["mean"],
                 frontier_topk_own=ev.topk_frontier(X, Y, tier.X_test, tier.Y_test, result["coef"], rel, folds,
                                                    ridge["fold_coefs"] if "ridge" in solver.name else None),
                 ridge_topk_jaccard_halves=ev.ridge_halves_stability(X, Y, rel, halves))
        return S, {}, time.perf_counter() - t0
    fold_paths, path = result.get("fold_paths"), result.get("path_coefs")
    recomputed = fold_paths is None or path is None
    if recomputed:
        fold_paths = [solver.fit_path(np.ascontiguousarray(X[tr]), np.ascontiguousarray(Y[tr]), alphas).coefs
                      for tr, va in folds]
        path = solver.fit_path(X, Y, alphas).coefs
    half_paths = [solver.fit_path(np.ascontiguousarray(X[h]), np.ascontiguousarray(Y[h]), alphas).coefs
                  for h in halves]
    S, per = ev.cv_analysis(tier, alphas, fold_paths, path, folds, half_paths, ridge)
    S["paths_recomputed"] = recomputed
    S["ridge_topk_jaccard_halves"] = ev.ridge_halves_stability(X, Y, rel, halves)
    return S, per, time.perf_counter() - t0


def _small(info, limit=2000):
    """Keep only JSON-friendly, small entries of a solver info dict."""
    out = {}
    for k, v in (info or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, (list, tuple, np.ndarray)) and np.size(v) <= limit:
            try:
                out[k] = np.asarray(v).tolist()
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(v, dict):
            out[k] = _small(v, limit)
    return out


def warmup(solver, tier, alphas, mode="small"):
    """Untimed compile/warm-up run on a slice of targets/alphas. Returns seconds spent (logged, excluded)."""
    if mode == "none":
        return None
    T = min(4, tier.Y.shape[1])
    A = min(5, alphas.shape[0])
    t0 = time.perf_counter()
    solver.fit_path(np.asarray(tier.X), np.ascontiguousarray(tier.Y[:, :T]), np.ascontiguousarray(alphas[:A, :T]))
    _sync(solver)
    return time.perf_counter() - t0


def run(solver_name, tier_name, task="P", solver_args=None, timeout=None, warmup_mode="small",
        max_targets=None, evaluate=True, eval_alpha_stride=1, out=RESULTS, tag=None, verbose=True,
        analysis=False):
    """Run one (solver, tier, task) and append a row to results.jsonl. Returns the row."""
    solver_args = dict(solver_args or {})
    solver = get_solver(solver_name, **solver_args)
    tier = load_tier(tier_name, max_targets=max_targets)
    t_load = time.perf_counter()
    X = np.asarray(tier.X)  # memmap -> page in once, outside timing (data loading is not the solver's job)
    Y = np.ascontiguousarray(tier.Y)
    t_load = time.perf_counter() - t_load
    alphas = api.alpha_grid(X, Y, N_ALPHAS, ALPHA_EPS)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    row = dict(run_id=run_id, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), task=task, solver=solver_name,
               tier=tier_name, shape=tier.shape, max_targets=max_targets, tag=tag,
               config=dict(tol=getattr(solver, "tol", None), max_iter=getattr(solver, "max_iter", None),
                           device=getattr(solver, "device", None), solver_args=solver_args,
                           n_alphas=N_ALPHAS, alpha_eps=ALPHA_EPS, n_folds=N_FOLDS if task == "CV" else None,
                           fold_seed=FOLD_SEED if task == "CV" else None, analysis=bool(analysis)),
               hardware=hardware_info(), **git_info(), time_load_s=t_load)
    if verbose:
        print(f"[run {run_id}] {solver_name} on {tier_name} {tier.shape} task {task}", flush=True)
    old = signal.signal(signal.SIGALRM, _alarm) if timeout else None
    try:
        try:
            row["time_warmup_s"] = warmup(solver, tier, alphas, warmup_mode) if hasattr(solver, "fit_path") and \
                type(solver).fit_path is not api.Solver.fit_path else None
        except NotImplementedError:
            row["time_warmup_s"] = None
        if timeout:
            signal.alarm(int(timeout))
        with MemSampler() as mem:
            if task == "P":
                t0 = time.perf_counter()
                t_prep = 0.0
                if hasattr(solver, "prepare"):
                    solver.prepare(X)
                    _sync(solver)
                    t_prep = time.perf_counter() - t0
                res, t_solve = _timed_fit_path(solver, X, Y, alphas)
                t_tot = time.perf_counter() - t0
                t_prep += float(res.info.get("time_prepare_s", 0.0)) if isinstance(res.info, dict) else 0.0
                result = res
            elif task == "CV":
                folds = make_folds(X.shape[0])
                t0 = time.perf_counter()
                if hasattr(solver, "fit_cv"):
                    cvr = solver.fit_cv(X, Y, alphas, folds)
                    _sync(solver)
                else:
                    cvr = default_fit_cv(solver, X, Y, alphas, folds, keep_paths=analysis)
                t_tot = time.perf_counter() - t0
                t_prep = float(cvr.get("info", {}).get("time_prepare_s", 0.0))
                t_solve = t_tot - t_prep
                result = cvr
            else:
                raise ValueError(task)
        if timeout:
            signal.alarm(0)
        row.update(status="ok", time_total_s=t_tot, time_prepare_s=t_prep, time_solve_s=t_tot - t_prep,
                   **mem.summary())
    except Timeout:
        row.update(status="timeout", time_total_s=float(timeout), timeout_s=float(timeout))
        result = None
    except Exception as e:  # noqa: BLE001
        if timeout:
            signal.alarm(0)
        row.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        result = None
    finally:
        if timeout:
            signal.signal(signal.SIGALRM, old)
    if verbose:
        print(f"[run {run_id}] status={row['status']} time_total={row.get('time_total_s')}", flush=True)

    raw = {}
    if result is not None and evaluate:
        ref = ev.load_reference(tier_name) if max_targets is None else None
        if ref is not None and (ref["objective"].shape != (alphas.shape[0], Y.shape[1])
                                or not np.allclose(ref["alphas"], alphas, rtol=1e-5)):
            print(f"[run {run_id}] cached reference does not match this tier's grid/targets; ignored", flush=True)
            ref = None
        t0 = time.perf_counter()
        if task == "P":
            check_path_result(result, X.shape[1], Y.shape[1], alphas.shape[0])
            rows = np.arange(0, alphas.shape[0], eval_alpha_stride)
            q = ev.path_quality(X, Y, result.coefs, alphas, ref=ref, alpha_rows=rows)
            row["metrics"] = ev.summarize_path(q)
            row["solver_info"] = _small(result.info)
            raw.update({k: q[k] for k in ("alpha_rows", "objective", "gap_rel", "nnz")})
            if "subopt_rel" in q:
                raw["subopt_rel"] = q["subopt_rel"]
        else:
            ba = result.get("best_idx")
            own_grid = result.get("alphas") is not None  # solver used its own grid (e.g. ridge): no Jaccard vs ref
            per, s = ev.cv_quality(tier, result["coef"], ba, result.get("alphas", alphas),
                                   ref=None if own_grid else ref)
            row["metrics"] = s
            row["solver_info"] = _small(result.get("info", {}))
            raw.update({k: v for k, v in per.items()})
            if result.get("cv_mse") is not None:
                raw["cv_mse"] = np.asarray(result["cv_mse"])
            if analysis:
                try:
                    A_s, A_per, A_t = run_analysis(solver, tier, X, Y, alphas, make_folds(X.shape[0]), result)
                    row["analysis"] = A_s
                    row["time_analysis_s"] = A_t
                    raw.update({f"an_{k}": v for k, v in A_per.items()})
                except Exception as e:  # noqa: BLE001
                    row["analysis_error"] = f"{type(e).__name__}: {e}"
                    row["analysis_traceback"] = traceback.format_exc()[-3000:]
        row["time_eval_s"] = time.perf_counter() - t0
        row["has_reference"] = ref is not None
    if raw:
        os.makedirs(RAW_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(RAW_DIR, f"{run_id}.npz"), **raw)
        row["raw"] = os.path.join(RAW_DIR, f"{run_id}.npz")
    append_row(row, out)
    if verbose and "metrics" in row:
        print(json.dumps(row["metrics"], indent=None), flush=True)
    return row, result


def append_row(row, out=RESULTS):
    import fcntl
    os.makedirs(os.path.dirname(out), exist_ok=True)
    line = json.dumps(ev._jsonable(row) if hasattr(ev, "_jsonable") else row, default=_default)
    with open(out, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_results(path=RESULTS):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows
