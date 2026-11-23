"""Dataset tiers for the Lasso benchmark (owned by bench-harness).

Every tier lives in `STORE/data/<tier>/` as float32 `X_train.npy, Y_train.npy, X_test.npy, Y_test.npy`
plus `meta.json`. X is z-scored with training-row statistics (constant columns dropped), Y is centered
and z-scored with training-row statistics. Load with `load_tier(name)`, which memory-maps the arrays.

Builders live here too (`build_tier(name)`); the heavy real-data tiers are built by
`scripts/build_unit_tiers.py` on a GPU node, which calls `save_tier`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from lassobench.api import STORE, make_synthetic

DATA_ROOT = os.path.join(STORE, "data")

# Raw sources (read-only)
VVS = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation"
PCA_PKL = (f"{VVS}/model_outputs/paul_20241119-20241122_Lasso/"
           "paul_20241119-20241122_resnet50_robust_deep_features_Xdict.pkl")
NEURAL_H5_PAUL = f"{VVS}/vvs-accentuate-day1_normalize_paul_241119-241122.h5"


@dataclass
class Tier:
    name: str
    X: np.ndarray        # (n_train, p) float32
    Y: np.ndarray        # (n_train, T) float32
    X_test: np.ndarray   # (n_test, p)
    Y_test: np.ndarray   # (n_test, T)
    meta: dict = field(default_factory=dict)
    path: str = ""

    @property
    def shape(self):
        return dict(n_train=int(self.X.shape[0]), n_test=int(self.X_test.shape[0]),
                    p=int(self.X.shape[1]), T=int(self.Y.shape[1]))

    def reliable_mask(self) -> np.ndarray:
        m = self.meta.get("reliable_mask")
        return np.ones(self.Y.shape[1], bool) if m is None else np.asarray(m, bool)[: self.Y.shape[1]]


def tier_dir(name: str) -> str:
    return os.path.join(DATA_ROOT, name)


def list_tiers() -> list[str]:
    if not os.path.isdir(DATA_ROOT):
        return []
    return sorted(d for d in os.listdir(DATA_ROOT) if os.path.exists(os.path.join(DATA_ROOT, d, "meta.json")))


def load_tier(name: str, mmap: bool = True, max_targets: int | None = None) -> Tier:
    """Load a built tier. Arrays are read-only memmaps unless mmap=False.

    max_targets: keep only the first k channels (for quick dev runs; recorded by the runner).
    """
    d = tier_dir(name)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise FileNotFoundError(f"tier {name!r} not built under {d}; available: {list_tiers()}")
    mode = "r" if mmap else None
    arr = {k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode=mode) for k in
           ("X_train", "Y_train", "X_test", "Y_test")}
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    Y, Yt = arr["Y_train"], arr["Y_test"]
    if max_targets is not None and max_targets < Y.shape[1]:
        Y, Yt = np.ascontiguousarray(Y[:, :max_targets]), np.ascontiguousarray(Yt[:, :max_targets])
    return Tier(name, arr["X_train"], Y, arr["X_test"], Yt, meta, d)


# ----------------------------------------------------------------------------- preprocessing

def standardize_X(X_train, X_test, eps=1e-6, chunk=65536):
    """z-score columns on training rows; drop (near-)constant columns. Returns (Xtr, Xte, keep, mu, sd)."""
    p = X_train.shape[1]
    mu = np.empty(p, np.float64)
    sd = np.empty(p, np.float64)
    for j in range(0, p, chunk):
        c = np.asarray(X_train[:, j:j + chunk], dtype=np.float64)
        mu[j:j + chunk] = c.mean(0)
        sd[j:j + chunk] = c.std(0)
    keep = sd > eps * max(1.0, float(np.median(sd[sd > 0])) if (sd > 0).any() else 1.0)
    idx = np.flatnonzero(keep)
    Xtr = np.empty((X_train.shape[0], idx.size), np.float32)
    Xte = np.empty((X_test.shape[0], idx.size), np.float32)
    for j in range(0, idx.size, chunk):
        cols = idx[j:j + chunk]
        Xtr[:, j:j + chunk] = (np.asarray(X_train[:, cols], np.float64) - mu[cols]) / sd[cols]
        Xte[:, j:j + chunk] = (np.asarray(X_test[:, cols], np.float64) - mu[cols]) / sd[cols]
    return Xtr, Xte, keep, mu, sd


def standardize_Y(Y_train, Y_test):
    mu = Y_train.mean(0, dtype=np.float64)
    sd = Y_train.std(0, dtype=np.float64)
    sd = np.where(sd > 0, sd, 1.0)
    return (((Y_train - mu) / sd).astype(np.float32), ((Y_test - mu) / sd).astype(np.float32), mu, sd)


def save_tier(name, X, Y, X_test, Y_test, meta: dict, overwrite=False):
    d = tier_dir(name)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "meta.json")) and not overwrite:
        raise FileExistsError(f"{d} exists; pass overwrite=True")
    for k, a in (("X_train", X), ("Y_train", Y), ("X_test", X_test), ("Y_test", Y_test)):
        if a is None:
            continue  # caller already wrote this file (chunked builders)
        np.save(os.path.join(d, f"{k}.npy"), np.ascontiguousarray(a, dtype=np.float32))
    meta = dict(meta)
    meta.setdefault("name", name)
    meta.setdefault("built", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(_jsonable(meta), f, indent=1)
    return d


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def old_split(n, test_size=0.2, seed=42):
    """The split used throughout the old repo: train_test_split(test_size=0.2, random_state=42)."""
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(n), test_size=test_size, random_state=seed, shuffle=True)
    return np.asarray(tr), np.asarray(te)


def load_paul_neural(subject="paul_20241119-20241122"):
    """Responses + per-channel metadata for the 2024-11 paul session (1028 images)."""
    import h5py
    with h5py.File(NEURAL_H5_PAUL, "r") as f:
        g = f[subject]
        R = g["repavg/response_peak"][()].astype(np.float64)
        names = [s.decode() if isinstance(s, bytes) else str(s) for s in g["repavg/stimulus_name"][()]]
        md = {k: g["neuron_metadata"][k][()] for k in g["neuron_metadata"].keys()}
    rel = np.asarray(md.get("reliability"), np.float64).ravel() if "reliability" in md else None
    ncsnr = np.asarray(md.get("ncsnr"), np.float64).ravel() if "ncsnr" in md else None
    area = md.get("brain_area")
    if area is not None:
        area = [a.decode() if isinstance(a, bytes) else str(a) for a in np.asarray(area).ravel()]
    return dict(resp=R, stim_names=names, reliability=rel, ncsnr=ncsnr, brain_area=area, subject=subject)


RELIABILITY_THRESH = 0.5


def neural_meta(nd, tr, te):
    rel = nd["reliability"]
    m = dict(subject=nd["subject"], neural_h5=NEURAL_H5_PAUL, n_images=len(nd["stim_names"]),
             split="train_test_split(test_size=0.2, random_state=42)",
             train_idx=tr, test_idx=te, reliability=rel, ncsnr=nd["ncsnr"], brain_area=nd["brain_area"])
    if rel is not None:
        m["reliable_mask"] = (rel >= RELIABILITY_THRESH)
        m["reliable_def"] = f"reliability >= {RELIABILITY_THRESH}"
    return m


# ----------------------------------------------------------------------------- builders

def build_syn_s(overwrite=False):
    cfg = dict(n=1000, p=20000, T=16, k=30, snr=1.0, rho=0.5, seed=0, n_test=500)
    d = make_synthetic(**cfg)
    meta = dict(source="make_synthetic", config=cfg)
    out = save_tier("syn-s", d["X"], d["Y"], d["X_test"], d["Y_test"], meta, overwrite)
    import scipy.sparse as sp
    sp.save_npz(os.path.join(out, "W_true.npz"), sp.csc_matrix(d["W_true"]))
    return out


def _pca1k_raw():
    import pickle
    with open(PCA_PKL, "rb") as f:
        Xdict, resp, fps = pickle.load(f)
    X = np.asarray(Xdict[(".layer4.Bottleneck0", "pca1000")], np.float64)
    return X, np.asarray(resp, np.float64), fps


def build_pca1k(overwrite=False):
    """Old dev-notebook features, standardized (the benchmark tier) and unstandardized (`pca1k-raw`)."""
    X, R, fps = _pca1k_raw()
    nd = load_paul_neural("paul_20241119-20241122")
    assert np.allclose(nd["resp"], R), "pickle responses do not match HDF5 subject"
    tr, te = old_split(len(X))
    Ytr, Yte, ymu, ysd = standardize_Y(R[tr], R[te])
    base = neural_meta(nd, tr, te)
    base.update(model="resnet50_robust", layer=".layer4.Bottleneck0", features="pca1000 (old pickle)",
                source=PCA_PKL, y_mean=ymu, y_std=ysd)
    Xtr, Xte, keep, mu, sd = standardize_X(X[tr], X[te])
    out = save_tier("pca1k", Xtr, Ytr, Xte, Yte, dict(base, x_standardized=True,
                    x_col_std_before=[float(sd.min()), float(sd.max())]), overwrite)
    # Unstandardized control: as the old repo used it (PCA scores, centered on all 1028 rows, std 1..48).
    # Columns centered on train rows so the no-intercept contract holds; scale left untouched.
    mu_tr = X[tr].mean(0)
    save_tier("pca1k-raw", (X[tr] - mu_tr), Ytr, (X[te] - mu_tr), Yte,
              dict(base, x_standardized=False, note="control tier for the standardization study only"),
              overwrite)
    return out


def build_syn_xl(n=10000, p=1_000_000, T=384, k=30, snr=1.0, rho=0.5, seed=0, n_test=1000,
                 name="syn-xl", chunk=20000, overwrite=False):
    """Same generative model as `make_synthetic`, but generated column-chunk by column-chunk into memmaps.

    Not bit-identical to make_synthetic (different RNG stream order), same distribution.
    """
    import scipy.sparse as sp
    d = tier_dir(name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    m = n + n_test
    n_fac = max(8, p // 50)
    F = rng.standard_normal((m, n_fac)).astype(np.float32)
    fac_of = rng.integers(0, n_fac, p)
    W = np.zeros((p, T), np.float32)
    for t in range(T):
        W[rng.choice(p, k, replace=False), t] = rng.standard_normal(k)
    Xtr = np.lib.format.open_memmap(os.path.join(d, "X_train.npy"), "w+", np.float32, (n, p))
    Xte = np.lib.format.open_memmap(os.path.join(d, "X_test.npy"), "w+", np.float32, (n_test, p))
    S = np.zeros((m, T), np.float64)
    for j in range(0, p, chunk):
        cols = slice(j, min(p, j + chunk))
        c = np.sqrt(rho) * F[:, fac_of[cols]] + np.sqrt(1 - rho) * rng.standard_normal((m, cols.stop - j), np.float32)
        mu, sd = c[:n].mean(0), c[:n].std(0)
        c = (c - mu) / sd
        Xtr[:, cols] = c[:n]
        Xte[:, cols] = c[n:]
        nzr = np.flatnonzero(np.any(W[cols] != 0, axis=1))
        if nzr.size:
            S += c[:, nzr].astype(np.float64) @ W[cols][nzr].astype(np.float64)
    Xtr.flush(); Xte.flush()
    del Xtr, Xte
    Yall = S + rng.standard_normal(S.shape) * S[:n].std(0) / np.sqrt(snr)
    Ytr, Yte, ymu, ysd = standardize_Y(Yall[:n], Yall[n:])
    cfg = dict(n=n, p=p, T=T, k=k, snr=snr, rho=rho, seed=seed, n_test=n_test)
    save_tier(name, None, Ytr, None, Yte, dict(source="build_syn_xl (chunked make_synthetic)", config=cfg),
              overwrite=True)
    sp.save_npz(os.path.join(d, "W_true.npz"), sp.csc_matrix(W / ysd[None, :].astype(np.float32)))
    return d


BUILDERS = {"syn-s": build_syn_s, "pca1k": build_pca1k, "pca1k-raw": build_pca1k, "syn-xl": build_syn_xl}


def build_tier(name, **kw):
    if name not in BUILDERS:
        raise KeyError(f"no in-process builder for {name!r}; real unit tiers: scripts/build_unit_tiers.py")
    return BUILDERS[name](**kw)
