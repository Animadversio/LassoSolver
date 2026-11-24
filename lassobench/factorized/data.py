"""Data for factorized readouts: a tier's X reshaped to feature maps, the raw (un-normalized) maps, raw responses.

All three views share the tier's rows (meta train_idx / test_idx), so everything is evaluated on the tier's
test split in the tier's Y units (z-scored on training rows), the same numbers Ridge / Lasso rows report.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from lassobench.api import STORE
from lassobench.data import load_tier

FEAT_DIR = os.path.join(STORE, "cache", "features")
OUT_DIR = os.path.join(STORE, "results", "factorized")

# layer -> (feature-map shape, raw-feature cache file)
LAYER_MAPS = {
    ".layer4.Bottleneck0": ((2048, 7, 7), "resnet50_robust_layer4.B0_paul1119-1122.npy"),
    ".layer3.Bottleneck5": ((1024, 14, 14), "resnet50_robust_layer3.B5_paul1119-1122.npy"),
    ".layer3.Bottleneck0": ((1024, 14, 14), "resnet50_robust_layer3.B0_paul1119-1122.npy"),
    ".layer1.Bottleneck0": ((256, 56, 56), "resnet50_robust_layer1.B0_paul1119-1122.npy"),
}


@dataclass
class MapData:
    tier: str
    shape: tuple            # (C, H, W)
    Xz: np.ndarray          # (n_train, C, H, W) tier X (z-scored), dropped constant columns re-inserted as 0
    Xz_test: np.ndarray
    Y: np.ndarray           # (n_train, T) tier Y (z-scored)
    Y_test: np.ndarray
    y_mean: np.ndarray      # tier Y = (R - y_mean) / y_std
    y_std: np.ndarray
    reliable: np.ndarray    # (T,) bool
    meta: dict
    Xraw: np.ndarray | None = None       # (n_train, C, H, W) un-normalized maps (for BatchNorm models)
    Xraw_test: np.ndarray | None = None
    R: np.ndarray | None = None          # (n_train, T) raw trial-averaged responses (repavg response_peak)
    R_test: np.ndarray | None = None

    @property
    def p(self):
        return int(np.prod(self.shape))

    def to_tier_units(self, pred_raw):
        """Raw-response-scale predictions -> tier Y units."""
        return (pred_raw - self.y_mean) / self.y_std


def _full(Xk, keep, p):
    if Xk.shape[1] == p:
        return np.asarray(Xk, np.float32)
    out = np.zeros((Xk.shape[0], p), np.float32)
    out[:, keep] = Xk
    return out


def load_maps(tier_name: str, raw: bool = True) -> MapData:
    t = load_tier(tier_name)
    m = t.meta
    shape, fname = LAYER_MAPS[m["layer"]]
    p = int(np.prod(shape))
    kp = os.path.join(t.path or os.path.join(STORE, "data", tier_name), "kept_columns.npy")
    keep = np.load(kp) if os.path.exists(kp) else np.arange(p)
    Xz = _full(np.asarray(t.X), keep, p).reshape(-1, *shape)
    Xz_test = _full(np.asarray(t.X_test), keep, p).reshape(-1, *shape)
    d = MapData(tier=tier_name, shape=shape, Xz=Xz, Xz_test=Xz_test, Y=np.asarray(t.Y, np.float32),
                Y_test=np.asarray(t.Y_test, np.float32), y_mean=np.asarray(m["y_mean"]),
                y_std=np.asarray(m["y_std"]), reliable=t.reliable_mask(), meta=m)
    if raw:
        tr, te = np.asarray(m["train_idx"]), np.asarray(m["test_idx"])
        F = np.load(os.path.join(FEAT_DIR, fname), mmap_mode="r")
        d.Xraw = np.asarray(F[tr], np.float32).reshape(-1, *shape)
        d.Xraw_test = np.asarray(F[te], np.float32).reshape(-1, *shape)
        from lassobench.data import load_paul_neural
        nd = load_paul_neural(m.get("subject", "paul_20241119-20241122"))
        d.R, d.R_test = nd["resp"][tr].astype(np.float32), nd["resp"][te].astype(np.float32)
        # consistency: the tier's Y is the z-scored raw response
        assert np.allclose(d.to_tier_units(d.R), d.Y, atol=1e-4), "raw responses do not match tier Y"
    return d


def member_split(n: int, seed: int, val_fraction: float = 0.2):
    """dualneuron's split_train_val: RandomState(seed) permutation, first round(n*f) rows are validation."""
    perm = np.random.RandomState(seed).permutation(n)
    n_val = int(round(n * val_fraction))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
