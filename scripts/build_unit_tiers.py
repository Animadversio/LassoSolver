#!/usr/bin/env python
"""Build real unit-basis tiers on a GPU node (owned by bench-harness).

  unit-100k : resnet50_robust .layer4.Bottleneck0 output (2048x7x7 = 100352), paul_20241119-20241122, 64 ch
  unit-800k : resnet50_robust .layer1.Bottleneck0 output (256x56x56 = 802816), paul_241119-241122, 383 ch
              (same 1028 images; falls back to the 64-ch subject if the stimulus lists differ)

Images and order come from the old Xdict pickle (whose responses equal the HDF5 subject's repavg exactly);
split = old train_test_split(test_size=0.2, random_state=42). X z-scored on training rows, constant columns
dropped, Y z-scored on training rows. Raw features cached under STORE/cache/features/.

Sanity check: the top principal components of our layer4.B0 features must match the pickle's pca1000 columns
(same images, same order, same model) -> |corr| ~ 1.

Usage: python scripts/build_unit_tiers.py [unit-100k] [unit-800k] [unit-100k-raw] [unit-800k-raw] [--overwrite]
"-raw" tiers (spec v0.1): same features/rows/columns kept, centered on train rows, one global scale.
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD = "/n/home12/binxuwang/Github/Closed-loop-visual-insilico"
sys.path.insert(1, OLD)

from lassobench.api import STORE  # noqa: E402
from lassobench.data import (PCA_PKL, load_paul_neural, neural_meta, old_split, save_tier, standardize_X,  # noqa: E402
                             standardize_Y, tier_dir)

FEAT_DIR = os.path.join(STORE, "cache", "features")
LAYERS = {"unit-100k": ("layer4", 0, "paul_20241119-20241122"),
          "unit-800k": ("layer1", 0, "paul_241119-241122")}
LAYERS.update({f"{k}-raw": v for k, v in list(LAYERS.items())})


def extract(fps, layers, batch_size=64):
    import torch
    from circuit_toolkit.dataset_utils import ImagePathDataset
    from core.model_load_utils import load_model_transform
    from torch.utils.data import DataLoader
    model, tfm = load_model_transform("resnet50_robust", device="cuda")
    model = model.cuda().eval()
    feats = {k: [] for k in layers}
    hooks = []
    for (stage, blk) in layers:
        mod = getattr(model, stage)[blk]
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, key=(stage, blk): feats[key].append(o.detach().float().flatten(1).cpu().numpy())))
    loader = DataLoader(ImagePathDataset(fps, scores=None, transform=tfm), batch_size=batch_size, shuffle=False,
                        num_workers=8)
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.cuda())
    for h in hooks:
        h.remove()
    return {k: np.concatenate(v, 0) for k, v in feats.items()}


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(LAYERS)
    ow = "--overwrite" in sys.argv
    with open(PCA_PKL, "rb") as f:
        Xdict, resp, fps = pickle.load(f)
    os.makedirs(FEAT_DIR, exist_ok=True)
    need = {LAYERS[n][:2] for n in names} | {("layer4", 0)}
    raw = {}
    todo = []
    for k in need:
        fp = os.path.join(FEAT_DIR, f"resnet50_robust_{k[0]}.B{k[1]}_paul1119-1122.npy")
        if os.path.exists(fp):
            raw[k] = np.load(fp, mmap_mode="r")
        else:
            todo.append(k)
    if todo:
        t0 = time.time()
        out = extract(fps, todo)
        print(f"extracted {[(k, v.shape) for k, v in out.items()]} in {time.time() - t0:.1f}s", flush=True)
        for k, v in out.items():
            np.save(os.path.join(FEAT_DIR, f"resnet50_robust_{k[0]}.B{k[1]}_paul1119-1122.npy"), v)
            raw[k] = v

    # sanity: our layer4.B0 vs pickle pca1000
    F = np.asarray(raw[("layer4", 0)], np.float64)
    Fc = F - F.mean(0)
    U, S, _ = np.linalg.svd(Fc @ Fc.T)
    P = np.asarray(Xdict[(".layer4.Bottleneck0", "pca1000")], np.float64)
    pc_corr = [abs(np.corrcoef(U[:, i], P[:, i])[0, 1]) for i in range(5)]
    print("sanity |corr(our PC_i, pickle pca_i)| i<5:", np.round(pc_corr, 4), flush=True)

    tr, te = old_split(len(fps))
    stim_names = [os.path.basename(f) for f in fps]
    for name in names:
        stage, blk, subj = LAYERS[name]
        nd = load_paul_neural(subj)
        if nd["stim_names"] != stim_names:
            print(f"{subj} stimulus list differs from pickle; falling back to paul_20241119-20241122", flush=True)
            nd = load_paul_neural("paul_20241119-20241122")
        R = nd["resp"]
        X = raw[(stage, blk)]
        t0 = time.time()
        raw_tier = name.endswith("-raw")
        Xtr, Xte, keep, mu, sd = standardize_X(X[tr], X[te], per_column=not raw_tier)
        Ytr, Yte, ymu, ysd = standardize_Y(R[tr], R[te])
        meta = neural_meta(nd, tr, te)
        meta.update(model="resnet50_robust", layer=f".{stage}.Bottleneck{blk}", features="unit basis (flattened C,H,W)",
                    n_features_raw=int(X.shape[1]), n_features_kept=int(keep.sum()),
                    n_constant_dropped=int((~keep).sum()), image_source=PCA_PKL, y_mean=ymu, y_std=ysd,
                    sanity_pc_corr_layer4B0=pc_corr, x_standardized=not raw_tier,
                    x_scaling=("centered, one global scale %.6g" % sd[keep][0]) if raw_tier else "z-score per column", frac_zero_entries=float(np.mean(np.asarray(X[tr]) == 0)))
        save_tier(name, Xtr, Ytr, Xte, Yte, meta, overwrite=ow)
        np.save(os.path.join(tier_dir(name), "kept_columns.npy"), np.flatnonzero(keep))
        print(f"{name}: X {Xtr.shape} (dropped {(~keep).sum()} constant), Y {Ytr.shape}, "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
