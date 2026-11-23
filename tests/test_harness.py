"""Fast harness tests on syn-s (a few targets). Run: python -m pytest tests -q"""
import json
import os

import numpy as np
import pytest
import scipy.sparse as sp

from lassobench import api, evaluate as ev, runner
from lassobench.data import build_tier, list_tiers, load_tier


@pytest.fixture(scope="module")
def tier():
    if "syn-s" not in list_tiers():
        build_tier("syn-s")
    return load_tier("syn-s", max_targets=3)


def test_tier_shapes(tier):
    assert tier.X.dtype == np.float32 and tier.X.shape == (1000, 20000)
    assert tier.Y.shape == (1000, 3) and tier.X_test.shape[1] == tier.X.shape[1]
    assert np.allclose(np.asarray(tier.X[:, :50]).mean(0), 0, atol=1e-4)
    assert np.allclose(np.asarray(tier.X[:, :50]).std(0), 1, atol=1e-3)


def test_discovery():
    s, _ = runner.discover_solvers()
    for k in ("cpu_sklearn_cd", "cpu_celer", "cpu_ridgecv"):
        assert k in s


def test_gap_matches_api(tier):
    X, Y = np.asarray(tier.X), np.asarray(tier.Y)
    al = api.alpha_grid(X, Y, 5)
    rng = np.random.default_rng(0)
    W = sp.random(X.shape[1], Y.shape[1], density=1e-3, random_state=0, format="csc") * 0.01
    q = ev.path_quality(X, Y, [W] * 5, al)
    for a in range(5):
        np.testing.assert_allclose(q["gap"][a], api.duality_gap(X.astype(np.float64), Y.astype(np.float64), W, al[a]),
                                   rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(q["objective"][a], api.objective(X.astype(np.float64), Y.astype(np.float64), W, al[a]),
                                   rtol=1e-8)


def test_zero_solution_at_alpha_max(tier):
    X, Y = np.asarray(tier.X), np.asarray(tier.Y)
    al = api.alpha_grid(X, Y, 3)
    Z = sp.csc_matrix((X.shape[1], Y.shape[1]))
    q = ev.path_quality(X, Y, [Z] * 3, al)
    assert np.all(np.abs(q["gap_rel"][0]) < 1e-5)  # w=0 optimal at alpha_max


@pytest.mark.parametrize("task", ["P", "CV"])
def test_run_appends_row(tmp_path, task):
    out = tmp_path / "r.jsonl"
    row, res = runner.run("cpu_sklearn_cd", "syn-s", task, dict(tol=1e-4 if task == "P" else 1e-2, n_jobs=3),
                          max_targets=2 if task == "P" else 1,
                          warmup_mode="none", out=str(out), verbose=False)
    assert row["status"] == "ok", row.get("traceback")
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 1 and rows[0]["run_id"] == row["run_id"]
    for k in ("time_total_s", "time_prepare_s", "time_solve_s", "peak_ram_gb", "hardware", "git_hash", "metrics"):
        assert k in rows[0]
    if task == "P":
        assert rows[0]["metrics"]["gap_rel_max"] < 3e-4
    else:
        assert rows[0]["metrics"]["r2_test_mean"] > 0.2
    os.remove(row["raw"])
