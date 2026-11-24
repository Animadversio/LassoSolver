"""Matched-sparsity comparison (spec sec 9): do gradient methods keep an R2 advantage at equal k?

Split: the tier's training rows are split 80/20 (seed 0) into fit / validation; every choice (alpha, ridge lambda,
IHT iteration count) is made per channel on the validation rows; the tier's test rows are only used for the
reported numbers. Metric: mean test R2 over reliable channels.

Methods (all fitted on the fit rows, alpha grid = alpha_grid(fit rows), 50 x 1e-2):
  lasso      exact path (torch_ws, tol 1e-4)
  prox_sgd   ISTA 1000 epochs; prox_adam 300 epochs lr 1e-2; hadamard u*v Adam 1000 epochs lr 1e-2 + final prox
  iht        accelerated projected gradient (step 1/L) with top-k projection every step, LS loss, 1000 iters
  iht_adam   Adam (lr 1e-2) with top-k projection every step, 1000 iters
Readouts at k in {10, 30, 100, 300, 1000} nonzeros per channel:
  trunc      top-k |w| of the path solution at each alpha; alpha chosen on validation
  refit      top-k support then ridge refit on the kept units (lambda grid incl. ~0); (alpha, lambda) on validation
  path       (lasso only) the path solution whose nnz is closest to k, no truncation; refit on its support
Reference: ridge on all features (lambda chosen on validation).

usage: python scripts/dev_prox_sparsity.py [--tier unit-100k] [--out STORE/results/prox/sparsity_<tier>.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lassobench.api import STORE, alpha_grid  # noqa: E402
from lassobench.data import load_tier  # noqa: E402
from lassobench.solvers._torch_common import device_name, lipschitz  # noqa: E402

KS = [10, 30, 100, 300, 1000]
LAMS = np.concatenate([[1e-8], np.logspace(-4, 2, 13)])   # ridge lambda (objective ||y-Xw||^2/(2n) + lam/2 ||w||^2)
dev = torch.device("cuda")
DT = torch.float64


def r2(Yhat, Yt):
    """Per-channel R2, (..., n, T) predictions vs (n, T)."""
    sst = ((Yt - Yt.mean(0)) ** 2).sum(0)
    return 1 - ((Yt - Yhat) ** 2).sum(-2) / sst


def topk_cols(W, k):
    """Keep the k largest |w| per column of W (p, T) (all nonzeros if fewer)."""
    if k >= W.shape[0]:
        return W
    idx = W.abs().topk(k, dim=0).indices
    out = torch.zeros_like(W)
    return out.scatter_(0, idx, W.gather(0, idx))


def ridge_refit(Xa, ya, Xv, yv, Xt, yt, S):
    """Ridge on columns S for each lambda; returns (val R2, test R2), each (C, n_lambda). S: list of C non-empty
    index tensors of equal length (candidate supports). Batched SVD over candidates."""
    Si = torch.stack(S)                                     # (C, k)
    Xs = Xa[:, Si].permute(1, 0, 2)                         # (C, n, k)
    n = Xa.shape[0]
    U, s, Vh = torch.linalg.svd(Xs, full_matrices=False)    # (C,n,r),(C,r),(C,r,k)
    uy = U.transpose(1, 2) @ ya                             # (C, r)
    lam = torch.as_tensor(LAMS, device=dev, dtype=DT)
    f = s[:, None, :] / (s[:, None, :] ** 2 + n * lam[None, :, None])        # (C, L, r)
    Wc = torch.einsum("clr,cr,crk->clk", f, uy, Vh)        # (C, L, k)
    pv = torch.einsum("cvk,clk->clv", Xv[:, Si].permute(1, 0, 2), Wc)
    pt = torch.einsum("cvk,clk->clv", Xt[:, Si].permute(1, 0, 2), Wc)
    def r2v(P, y):
        sst = ((y - y.mean()) ** 2).sum()
        return 1 - ((y - P) ** 2).sum(-1) / sst
    return r2v(pv, yv), r2v(pt, yt)                         # (C, L) each


def best_refit(Xa, ya, Xv, yv, Xt, yt, cands):
    """Best (val-chosen) ridge-refit test R2 over candidate supports of any sizes (grouped by size for batching)."""
    cands = [c for c in cands if len(c) > 0]
    if not cands:
        return 0.0
    best_v, best_t = -1e9, 0.0
    by = {}
    for c in cands:
        by.setdefault(len(c), []).append(c)
    for group in by.values():
        vv, vt = ridge_refit(Xa, ya, Xv, yv, Xt, yt, group)
        j = int(vv.argmax())
        if float(vv.flatten()[j]) > best_v:
            best_v, best_t = float(vv.flatten()[j]), float(vt.flatten()[j])
    return best_t


def fit_methods(Xa_h, Ya_h, alphas, alphas_ext):
    from lassobench.solvers.torch_proxgrad import SOLVERS as PG
    from lassobench.solvers.torch_ws import SOLVERS as WS
    cfg = {
        "lasso": (WS["torch_ws"], dict(tol=1e-4)),
        "lasso_ext": (WS["torch_ws"], dict(tol=1e-4)),          # same, grid extended to 1e-3 alpha_max
        "prox_sgd": (PG["torch_prox_sgd"], dict(epochs=1000, lr_schedule="const")),
        "prox_adam": (PG["torch_prox_adam"], dict(epochs=300, lr=1e-2)),
        "hadamard": (PG["torch_hadamard_adam"], dict(epochs=1000, lr=1e-2, finalize="prox")),
    }
    out, times = {}, {}
    for m, (cls, kw) in cfg.items():
        t0 = time.perf_counter()
        al = alphas_ext if m == "lasso_ext" else alphas
        res = cls(**kw).fit_path(Xa_h, Ya_h, al)
        torch.cuda.synchronize()
        times[m] = time.perf_counter() - t0
        out[m] = torch.stack([torch.as_tensor(c.toarray(), device=dev, dtype=DT) for c in res.coefs])  # (A,p,T)
        print(f"fit {m}: {times[m]:.1f}s", flush=True)
        del res
    return out, times


def iht(Xa, Ya, ks, n_iter=1000, snaps=(100, 300, 1000), adam=False, lr=1e-2):
    """Batched top-k projected gradient. Returns {k: {iter: W (p, T)}}. LS loss 1/(2n)||y-Xw||^2."""
    n, p = Xa.shape
    T = Ya.shape[1]
    Xf = Xa.float()
    L = lipschitz(Xf)
    K = len(ks)
    Yb = Ya.float().T.repeat(K, 1)                          # (K*T, n)
    kk = torch.as_tensor(np.repeat(ks, T), device=dev)
    W = torch.zeros(K * T, p, device=dev)
    Wp = W.clone()
    m1 = torch.zeros_like(W); m2 = torch.zeros_like(W)
    tk = 1.0
    snapsd = {k: {} for k in ks}
    kmax = max(ks)
    rank = torch.arange(kmax, device=dev)[None, :]

    def project(V):
        vals, idx = V.abs().topk(kmax, dim=1)
        keep = rank < kk[:, None]
        out = torch.zeros_like(V)
        out.scatter_(1, idx, torch.where(keep, V.gather(1, idx), torch.zeros_like(vals)))
        return out

    for it in range(1, n_iter + 1):
        if adam:
            G = ((W @ Xf.T) - Yb) @ Xf / n
            m1.mul_(0.9).add_(G, alpha=0.1); m2.mul_(0.999).addcmul_(G, G, value=0.001)
            W = project(W - lr * (m1 / (1 - 0.9 ** it)) / ((m2 / (1 - 0.999 ** it)).sqrt() + 1e-8))
        else:
            t_next = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            Z = W + ((tk - 1) / t_next) * (W - Wp)
            G = ((Z @ Xf.T) - Yb) @ Xf / n
            Wp, W = W, project(Z - G / L)
            tk = t_next
        if it in snaps:
            for i, k in enumerate(ks):
                snapsd[k][it] = W[i * T:(i + 1) * T].T.to(DT).clone()
    return snapsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="unit-100k")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = load_tier(a.tier)
    X = np.asarray(t.X); Y = np.asarray(t.Y)
    rel = t.reliable_mask()
    n = X.shape[0]
    perm = np.random.default_rng(0).permutation(n)
    nv = n // 5
    iv, ia = np.sort(perm[:nv]), np.sort(perm[nv:])
    Xa_h, Ya_h = np.ascontiguousarray(X[ia]), np.ascontiguousarray(Y[ia])
    alphas = alpha_grid(Xa_h, Ya_h, 50, 1e-2)
    g = lambda A: torch.as_tensor(np.asarray(A), device=dev, dtype=DT)  # noqa: E731
    Xa, Ya, Xv, Yv, Xt, Yt = g(Xa_h), g(Ya_h), g(X[iv]), g(Y[iv]), g(t.X_test), g(t.Y_test)
    T = Y.shape[1]
    relt = torch.as_tensor(rel, device=dev)
    mean_rel = lambda v: float(v[relt].mean())  # noqa: E731
    print(f"# {a.tier} fit {len(ia)} val {nv} test {Xt.shape[0]} p {X.shape[1]} T {T} reliable {rel.sum()} "
          f"gpu {device_name()}", flush=True)
    res = dict(tier=a.tier, gpu=device_name(), ks=KS, n_fit=len(ia), n_val=nv, n_reliable=int(rel.sum()))

    # ridge reference on all features (kernel form), lambda on validation
    K = Xa @ Xa.T
    ev, Q = torch.linalg.eigh(K)
    lam = torch.as_tensor(np.logspace(-3, 4, 29), device=dev, dtype=DT)
    na = Xa.shape[0]
    qy = Q.T @ Ya                                            # (na, T)
    coef = qy[None] / (ev[None, :, None] + na * lam[:, None, None])       # (L, na, T)
    dual = Q[None] @ coef                                    # (L, na, T)
    rv = r2((Xv @ Xa.T)[None] @ dual, Yv); rt = r2((Xt @ Xa.T)[None] @ dual, Yt)
    bi = rv.argmax(0)
    res["ridge_test_r2_rel"] = mean_rel(rt[bi, torch.arange(T)])
    rt_ridge = rt[bi, torch.arange(T)][torch.as_tensor(rel, device=dev)].tolist()
    print("ridge (all features):", res["ridge_test_r2_rel"], flush=True)

    alphas_ext = alpha_grid(Xa_h, Ya_h, 70, 1e-3)
    paths, times = fit_methods(Xa_h, Ya_h, alphas, alphas_ext)
    per = {}   # per-channel test R2 on reliable channels: per[(method, k, readout)] = list
    res["fit_times_s"] = times
    ar = torch.arange(T, device=dev)
    table = {}
    for m, Wp in paths.items():                              # Wp (A, p, T)
        nnz = (Wp != 0).sum(1)                               # (A, T)
        table[m] = {}
        for k in KS:
            Wk = torch.stack([topk_cols(Wp[i], k) for i in range(Wp.shape[0])])      # (A,p,T)
            rv = r2(Xv @ Wk, Yv); rt = r2(Xt @ Wk, Yt)                               # (A, T)
            ba = rv.argmax(0)
            row = dict(trunc=mean_rel(rt[ba, ar]),
                       trunc_nnz=float(torch.minimum(nnz[ba, ar], torch.tensor(k, device=dev)).float()[relt].median()))
            # refit: (alpha, lambda) on validation over the top-k supports of every alpha
            best_t = torch.zeros(T, device=dev, dtype=DT)
            for tt in range(T):
                cands = [torch.nonzero(Wk[i, :, tt])[:, 0] for i in range(Wk.shape[0])]
                best_t[tt] = best_refit(Xa, Ya[:, tt], Xv, Yv[:, tt], Xt, Yt[:, tt], cands)
            per[(m, k, "trunc")] = rt[ba, ar][relt].tolist()
            per[(m, k, "refit")] = best_t[relt].tolist()
            row["refit"] = mean_rel(best_t)
            if m.startswith("lasso"):
                ia_k = (nnz - k).abs().argmin(0)                                    # path read at nnz ~ k
                row["path"] = mean_rel(r2(Xt @ Wp[ia_k, :, ar].T, Yt))
                row["path_nnz"] = float(nnz[ia_k, ar].float()[relt].median())
                pr = torch.zeros(T, device=dev, dtype=DT)
                for tt in range(T):
                    S = torch.nonzero(Wp[ia_k[tt], :, tt])[:, 0]
                    pr[tt] = best_refit(Xa, Ya[:, tt], Xv, Yv[:, tt], Xt, Yt[:, tt], [S])
                row["path_refit"] = mean_rel(pr)
                per[(m, k, "path")] = r2(Xt @ Wp[ia_k, :, ar].T, Yt)[relt].tolist()
                per[(m, k, "path_refit")] = pr[relt].tolist()
            table[m][k] = row
            print(m, k, {kk: round(v, 4) for kk, v in row.items()}, flush=True)
    for name, adam in (("iht", False), ("iht_adam", True)):
        t0 = time.perf_counter()
        sn = iht(Xa, Ya, KS, adam=adam)
        torch.cuda.synchronize()
        times[name] = time.perf_counter() - t0
        table[name] = {}
        for k in KS:
            Ws = torch.stack(list(sn[k].values()))                                  # (S, p, T)
            rv = r2(Xv @ Ws, Yv); rt = r2(Xt @ Ws, Yt)
            bs = rv.argmax(0)
            row = dict(trunc=mean_rel(rt[bs, ar]))
            best_t = torch.zeros(T, device=dev, dtype=DT)
            for tt in range(T):
                cands = [torch.nonzero(Ws[i, :, tt])[:, 0] for i in range(Ws.shape[0])]
                best_t[tt] = best_refit(Xa, Ya[:, tt], Xv, Yv[:, tt], Xt, Yt[:, tt], cands)
            row["refit"] = mean_rel(best_t)
            per[(name, k, "trunc")] = rt[bs, ar][relt].tolist()
            per[(name, k, "refit")] = best_t[relt].tolist()
            table[name][k] = row
            print(name, k, {kk: round(v, 4) for kk, v in row.items()}, f"({times[name]:.1f}s)", flush=True)
    res["table"] = table
    res["per_channel"] = {f"{m}|{k}|{r}": v for (m, k, r), v in per.items()}
    res["ridge_per_channel"] = rt_ridge
    out = a.out or str(Path(STORE) / "results" / "prox" / f"sparsity_{a.tier}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("# wrote", out)


if __name__ == "__main__":
    main()
