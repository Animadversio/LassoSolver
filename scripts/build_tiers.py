#!/usr/bin/env python
"""Build in-process tiers: syn-s, pca1k (+ pca1k-raw), syn-xl.  Usage: build_tiers.py syn-s pca1k [--overwrite]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lassobench.data import build_tier, load_tier  # noqa: E402

if __name__ == "__main__":
    ow = "--overwrite" in sys.argv
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        t0 = time.time()
        build_tier(name, overwrite=ow)
        t = load_tier(name)
        print(name, t.shape, f"{time.time() - t0:.1f}s", flush=True)
