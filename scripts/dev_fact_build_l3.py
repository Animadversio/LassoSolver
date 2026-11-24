#!/usr/bin/env python
"""factorized-readout: cache resnet50_robust layer3 feature maps and build tier `unit-200k-l3` (GPU node).

  layer3.B5 = output of resnet50_robust `layer3` (1024 x 14 x 14 = 200704 units), same images / order / split /
  channels as unit-100k. Raw (un-normalized) maps cached to STORE/cache/features/ (the Gaussian readout applies
  BatchNorm itself); the tier itself is built exactly like scripts/build_unit_tiers.py builds unit-100k
  (per-column z-score on train rows, constant columns dropped, Y z-scored on train rows).

Usage: python scripts/dev_fact_build_l3.py [--block 5] [--overwrite]
"""
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from build_unit_tiers import FEAT_DIR, extract  # noqa: E402  (read-only reuse of the harness extractor)
from lassobench.data import (PCA_PKL, load_paul_neural, neural_meta, old_split, save_tier, standardize_X,  # noqa: E402
                             standardize_Y, tier_dir)


def main():
    blk = int(sys.argv[sys.argv.index("--block") + 1]) if "--block" in sys.argv else 5
    ow = "--overwrite" in sys.argv
    name = "unit-200k-l3" if blk == 5 else f"unit-l3b{blk}"
    with open(PCA_PKL, "rb") as f:
        _, _, fps = pickle.load(f)
    fp = os.path.join(FEAT_DIR, f"resnet50_robust_layer3.B{blk}_paul1119-1122.npy")
    if os.path.exists(fp):
        X = np.load(fp, mmap_mode="r")
    else:
        t0 = time.time()
        X = extract(fps, [("layer3", blk)])[("layer3", blk)]
        print(f"extracted layer3.B{blk} {X.shape} in {time.time() - t0:.1f}s", flush=True)
        np.save(fp, X)
    tr, te = old_split(len(fps))
    nd = load_paul_neural("paul_20241119-20241122")
    assert nd["stim_names"] == [os.path.basename(f) for f in fps]
    Xtr, Xte, keep, mu, sd = standardize_X(X[tr], X[te], per_column=True)
    Ytr, Yte, ymu, ysd = standardize_Y(nd["resp"][tr], nd["resp"][te])
    meta = neural_meta(nd, tr, te)
    meta.update(model="resnet50_robust", layer=f".layer3.Bottleneck{blk}", features="unit basis (flattened C,H,W)",
                feature_map_shape=[1024, 14, 14], n_features_raw=int(X.shape[1]), n_features_kept=int(keep.sum()),
                n_constant_dropped=int((~keep).sum()), image_source=PCA_PKL, y_mean=ymu, y_std=ysd,
                x_standardized=True, x_scaling="z-score per column", raw_features=fp,
                built_by="factorized-readout (scripts/dev_fact_build_l3.py)",
                frac_zero_entries=float(np.mean(np.asarray(X[tr]) == 0)))
    save_tier(name, Xtr, Ytr, Xte, Yte, meta, overwrite=ow)
    np.save(os.path.join(tier_dir(name), "kept_columns.npy"), np.flatnonzero(keep))
    print(f"{name}: X {Xtr.shape} (dropped {(~keep).sum()} constant), Y {Ytr.shape}", flush=True)


if __name__ == "__main__":
    main()
