import numpy as np, sys
sys.path.insert(0, "/n/home12/binxuwang/Github/LassoSolver")
from lassobench.factorized.data import load_maps
for tier in ["unit-200k-l3", "unit-100k"]:
    d = load_maps(tier, raw=False)
    tr0, te0 = np.asarray(d.meta["train_idx"]), np.asarray(d.meta["test_idx"])
    N = len(tr0) + len(te0)
    Xall = np.empty((N, d.p), np.float32)
    Xall[tr0], Xall[te0] = d.Xz.reshape(len(tr0), -1), d.Xz_test.reshape(len(te0), -1)
    print(tier, "tier split: max|Xte|", np.abs(d.Xz_test).max(), "max|Xtr|", np.abs(d.Xz).max(), flush=True)
    nzfrac = (Xall != Xall.min(0)).mean(0)
    print("  columns by fraction of non-min images: <=1/1028:", (nzfrac <= 1/N).sum(), "<=5:", (nzfrac <= 5/N).sum(), "<=1%:", (nzfrac <= .01).sum())
    for k in range(1, 5):
        perm = np.random.RandomState(1000 + k).permutation(N)
        te, tr = np.sort(perm[:len(te0)]), np.sort(perm[len(te0):])
        mu, sd = Xall[tr].mean(0), Xall[tr].std(0)
        ok = sd > 1e-6
        Z = (Xall[te] - mu) / np.where(ok, sd, 1) * ok
        a = np.abs(Z).max(0)
        print(f"  split{k}: max|Xte| {a.max():.3g}; cols with max|Xte|>50: {(a > 50).sum()}, >1e3: {(a > 1e3).sum()}; "
              f"min sd among ok {sd[ok].min():.3g}; cols sd<1e-3: {((sd < 1e-3) & ok).sum()}", flush=True)
