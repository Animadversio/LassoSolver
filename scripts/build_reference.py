#!/usr/bin/env python
"""Reference Task P solutions per tier (float64, tight tol), cached at STORE/cache/ref/<tier>/.

Default: celer at tol=1e-9 (its gap is tighter than the harness rescaled-residual gap), then any
(alpha, target) whose harness gap_rel > --target-gap is re-solved with sklearn CD float64 at tol=target/2,
whose stopping gap is exactly the harness gap. The achieved gap is stored and reported.

Usage: python scripts/build_reference.py --tier pca1k [--solver cpu_celer] [--tol 1e-9] [--target-gap 1e-8]
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lassobench import api, evaluate as ev, runner  # noqa: E402
from lassobench.data import load_tier  # noqa: E402


def polish(X, Y, coefs, alphas, bad, target, max_iter):
    """Re-solve targets with any bad alpha using sklearn CD float64 over the full path."""
    from lassobench.solvers.cpu_sklearn_cd import SklearnCD
    ts = np.flatnonzero(bad.any(0))
    if ts.size == 0:
        return coefs, 0
    s = SklearnCD(tol=target, max_iter=max_iter, dtype="float64")
    res = s.fit_path(X, np.ascontiguousarray(Y[:, ts]), np.ascontiguousarray(alphas[:, ts]))
    pos = {int(t): i for i, t in enumerate(ts)}
    out = []
    for a, C in enumerate(coefs):
        C, Ca = sp.csc_matrix(C), sp.csc_matrix(res.coefs[a])
        out.append(sp.hstack([Ca[:, pos[t]] if t in pos else C[:, t] for t in range(C.shape[1])]).tocsc())
    return out, ts.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--solver", default="cpu_celer")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--target-gap", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=100000)
    ap.add_argument("--polish-rounds", type=int, default=2)
    a = ap.parse_args()
    tier = load_tier(a.tier)
    X, Y = np.asarray(tier.X), np.ascontiguousarray(tier.Y)
    alphas = api.alpha_grid(X, Y, runner.N_ALPHAS, runner.ALPHA_EPS)
    t0 = time.perf_counter()
    solver = runner.get_solver(a.solver, tol=a.tol, max_iter=a.max_iter, dtype="float64")
    res = solver.fit_path(X, Y, alphas)
    t_solve = time.perf_counter() - t0
    print(f"{a.solver} tol={a.tol}: {t_solve:.1f}s", flush=True)
    coefs = res.coefs
    q = ev.path_quality(X, Y, coefs, alphas)
    print("gap_rel max/median", q["gap_rel"].max(), np.median(q["gap_rel"]), flush=True)
    # provisional reference right away (polish can take hours at p >= 1e5 and jobs get preempted)
    base = dict(tier=a.tier, solver=a.solver, tol=a.tol, target_gap=a.target_gap, n_alphas=runner.N_ALPHAS,
                alpha_eps=runner.ALPHA_EPS, hardware=runner.hardware_info(), **runner.git_info(),
                solver_info=runner._small(res.info), time_first_solve_s=t_solve)
    ev.save_reference(a.tier, coefs, alphas, q, dict(base, stage="celer (unpolished)", time_total_s=t_solve,
                                                     gap_rel_max=float(q["gap_rel"].max()),
                                                     gap_rel_median=float(np.median(q["gap_rel"]))))
    print("saved provisional reference", flush=True)
    n_pol = []
    target = a.target_gap
    for r in range(a.polish_rounds):
        bad = q["gap_rel"] > a.target_gap
        if not bad.any():
            break
        t1 = time.perf_counter()
        coefs, k = polish(X, Y, coefs, alphas, bad, target / 2, a.max_iter)
        q = ev.path_quality(X, Y, coefs, alphas)
        n_pol.append(dict(round=r, targets=int(k), time_s=time.perf_counter() - t1, cd_tol=target / 2,
                          gap_rel_max=float(q["gap_rel"].max())))
        print("polish", n_pol[-1], flush=True)
        target /= 10
    t_tot = time.perf_counter() - t0
    info = dict(base, stage="polished" if n_pol else "celer", time_total_s=t_tot, polish=n_pol,
                gap_rel_max=float(q["gap_rel"].max()), gap_rel_median=float(np.median(q["gap_rel"])))
    ev.save_reference(a.tier, coefs, alphas, q, info)
    print(f"saved reference for {a.tier}: gap_rel max {info['gap_rel_max']:.3g}, total {t_tot:.1f}s", flush=True)


if __name__ == "__main__":
    main()
