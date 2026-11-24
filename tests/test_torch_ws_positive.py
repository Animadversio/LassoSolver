"""torch_ws with positive=True (nonnegative Lasso) against scikit-learn. Needs a CUDA GPU and cupy.

Run: python -m pytest tests/test_torch_ws_positive.py -q
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from sklearn.linear_model import Lasso  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from lassobench import api  # noqa: E402
from lassobench.solvers.torch_ws import TorchWS  # noqa: E402

TOL = 1e-5


@pytest.fixture(scope="module")
def data():
    d = api.make_synthetic(n=300, p=2000, T=4, k=10, n_test=50)
    X, Y = d["X"], d["Y"]
    return X, Y, api.alpha_grid(X, Y, 12, 1e-2)


def objective(X, y, w, alpha):
    r = y - X @ w
    return r @ r / (2 * len(y)) + alpha * np.abs(w).sum()


def gap_positive(X, y, w, alpha):
    """Duality gap of the nonnegative Lasso relative to ||y||^2 / (2n): one-sided dual constraint."""
    n = len(y)
    r = y - X @ w
    corr = max((X.T @ r).max() / n, 0.0)
    theta = r * min(1.0, alpha / max(corr, 1e-300))
    dual = (y @ y - (y - theta) @ (y - theta)) / (2 * n)
    return (objective(X, y, w, alpha) - dual) / (y @ y / (2 * n))


@pytest.mark.parametrize("kw", [{}, {"kernel": 1}, {"inner": "fista"}, {"polish": 2}], ids=str)
def test_path_is_the_nonnegative_lasso(data, kw):
    X, Y, alphas = data
    X64, Y64 = X.astype(np.float64), Y.astype(np.float64)
    res = TorchWS(tol=TOL, positive=True, **kw).fit_path(X, Y, alphas)
    free = TorchWS(tol=TOL).fit_path(X, Y, alphas)
    assert min(c.min() for c in free.coefs) < 0, "the unconstrained fit has no negative weight: test is vacuous"
    for a in range(alphas.shape[0]):
        W = res.coefs[a].toarray()
        assert W.min() >= 0
        for t in range(Y.shape[1]):
            assert gap_positive(X64, Y64[:, t], W[:, t], alphas[a, t]) <= TOL
    for a in (3, 7, 11):
        for t in range(Y.shape[1]):
            sk = Lasso(alpha=alphas[a, t], positive=True, fit_intercept=False, tol=1e-12, max_iter=200000)
            sk.fit(X64, Y64[:, t])
            ours = objective(X64, Y64[:, t], res.coefs[a].toarray()[:, t], alphas[a, t])
            ref = objective(X64, Y64[:, t], sk.coef_, alphas[a, t])
            assert ours - ref <= TOL * (Y64[:, t] @ Y64[:, t]) / (2 * len(Y64))


def test_cv_is_nonnegative_and_matches_the_default_cv(data):
    from lassobench import runner

    X, Y, alphas = data
    folds = [(tr, va) for tr, va in KFold(5, shuffle=True, random_state=0).split(np.arange(len(X)))]
    solver = TorchWS(tol=TOL, positive=True, cv_stop=None)
    cv = solver.fit_cv(X, Y, alphas, folds)
    assert cv["coef"].min() >= 0 and cv["coef"].nnz > 0
    ref = runner.default_fit_cv(solver, X, Y, alphas, folds)       # K + 1 separate fit_path calls
    assert np.array_equal(cv["best_idx"], ref["best_idx"])
    np.testing.assert_allclose(cv["cv_mse"], ref["cv_mse"], rtol=1e-3)


def test_positive_rejects_the_options_it_does_not_cover(data):
    X, Y, alphas = data
    folds = [(tr, va) for tr, va in KFold(5, shuffle=True, random_state=0).split(np.arange(len(X)))]
    with pytest.raises(ValueError):
        TorchWS(positive=True, l1_ratio=0.5).fit_path(X, Y, alphas)
    with pytest.raises(ValueError):
        TorchWS(positive=True, relax_gamma=0.01).fit_path(X, Y, alphas)
    with pytest.raises(ValueError):
        TorchWS(positive=True, relax="cv").fit_cv(X, Y, alphas, folds)
