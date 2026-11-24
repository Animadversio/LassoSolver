"""Conditioning diagnostic for FISTA on a tier: top eigenvalues of X X^T / n, and the Lipschitz constant of X
restricted to (a) the union over targets of the reference support at several alphas, (b) single-target supports.
usage: python scripts/dev_prox_spectrum.py unit-100k"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lassobench.data import load_tier  # noqa: E402
from lassobench.evaluate import ref_dir  # noqa: E402

tier = sys.argv[1]
t = load_tier(tier)
X = torch.as_tensor(np.asarray(t.X), device="cuda", dtype=torch.float64)
n, p = X.shape
K = X @ X.T / n
ev = torch.linalg.eigvalsh(K).flip(0).cpu().numpy()
print(f"{tier} n={n} p={p}  top eig of XX^T/n: {np.round(ev[:8], 1).tolist()}  10th {ev[9]:.1f}  50th {ev[49]:.2f}  "
      f"median {np.median(ev):.3f}  min {ev[-1]:.2e}")
T = t.Y.shape[1]
rf = Path(ref_dir(tier)) / "coefs.npz"
if rf.exists():
    C = sp.load_npz(rf).tocsc()
else:   # no reference yet: get supports from the GPU working-set solver
    from lassobench.api import alpha_grid
    from lassobench.solvers.torch_ws import SOLVERS as WS
    Xh, Yh = np.asarray(t.X), np.ascontiguousarray(t.Y)
    res = list(WS.values())[0](tol=1e-4).fit_path(Xh, Yh, alpha_grid(Xh, Yh, 50, 1e-2))
    C = sp.hstack([sp.csc_matrix(c) for c in res.coefs]).tocsc()
A = C.shape[1] // T
for a in [5, 15, 25, 35, 49]:
    Ca = C[:, a * T:(a + 1) * T]
    rows = np.unique(Ca.indices)
    Xs = X[:, torch.as_tensor(rows, device="cuda")]
    Ls = float(torch.linalg.eigvalsh(Xs @ Xs.T / n)[-1])
    singles = []
    for j in range(0, T, max(1, T // 8)):
        r = Ca[:, j].indices
        if len(r):
            Xj = X[:, torch.as_tensor(r, device="cuda")]
            singles.append(float(torch.linalg.eigvalsh(Xj.T @ Xj / n)[-1]))
    print(f"alpha idx {a}: union support {len(rows)} feats, L_union {Ls:.1f}; single-target supports "
          f"(median {int(np.median(np.diff(Ca.indptr)))} feats) L median {np.median(singles):.1f}")
