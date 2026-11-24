#!/usr/bin/env python
"""Run one solver on one tier and append a row to tables/results.jsonl.

  python scripts/run_bench.py --solver cpu_celer --tier syn-s --task P --solver-args tol=1e-6 n_jobs=8
  python scripts/run_bench.py --solver cpu_celer --tier pca1k --task P --tol-sweep 1e-2,1e-4,1e-6
  python scripts/run_bench.py --list
"""
import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver")
    ap.add_argument("--tier")
    ap.add_argument("--task", choices=["P", "CV"], default="P")
    ap.add_argument("--solver-args", nargs="*", default=[], help="k=v pairs passed to the solver constructor")
    ap.add_argument("--tol-sweep", default=None, help="comma list of tol values; one row per tol")
    ap.add_argument("--timeout", type=float, default=None, help="seconds; a timeout is recorded as a row")
    ap.add_argument("--warmup", choices=["none", "small"], default="small")
    ap.add_argument("--max-targets", type=int, default=None, help="dev only: first k channels")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-alpha-stride", type=int, default=1, help="evaluate gap on every k-th alpha (huge tiers)")
    ap.add_argument("--out", default=None, help="results file (default tables/results.jsonl)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--analysis", action="store_true", help="Task CV: round-2 frontier/stability/1-SE/relaxed refit (untimed)")
    ap.add_argument("--list", action="store_true", help="list solvers and tiers")
    a = ap.parse_args()

    from lassobench import runner
    from lassobench.data import list_tiers
    if a.list:
        s, e = runner.discover_solvers()
        print("solvers:", sorted(s))
        if e:
            print("import errors:", e)
        print("tiers:", list_tiers())
        return
    if not (a.solver and a.tier):
        ap.error("--solver and --tier are required")
    kw = parse_kv(a.solver_args)
    tols = [float(t) for t in a.tol_sweep.split(",")] if a.tol_sweep else [None]
    for tol in tols:
        if tol is not None:
            kw["tol"] = tol
        row, _ = runner.run(a.solver, a.tier, a.task, kw, timeout=a.timeout, warmup_mode=a.warmup,
                            max_targets=a.max_targets, evaluate=not a.no_eval,
                            eval_alpha_stride=a.eval_alpha_stride, out=a.out or runner.RESULTS, tag=a.tag,
                            analysis=a.analysis)
        if row["status"] == "error":
            print(row.get("traceback", ""), file=sys.stderr)


if __name__ == "__main__":
    main()
