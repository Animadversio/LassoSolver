"""Microbenchmark of the two FISTA matmuls, R(B,n)@X(n,p) and W(B,p)@X^T(p,n), across batch B and precision.

usage: python scripts/dev_prox_mmbench.py --n 1000 --p 100000
"""
import argparse
import json
import torch


def bench(fn, reps=20):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p", type=int, default=100000)
    ap.add_argument("--Bs", default="16,64,128,320,640,1280,3200")
    a = ap.parse_args()
    n, p = a.n, a.p
    print("#", torch.cuda.get_device_name(0), f"n={n} p={p}")
    X32 = torch.randn(n, p, device="cuda")
    for mode in ["fp32", "tf32", "bf16", "fp64"]:
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
        dt = {"fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}[mode]
        X = X32.to(dt)
        for B in map(int, a.Bs.split(",")):
            R = torch.randn(B, n, device="cuda", dtype=dt)
            W = torch.randn(B, p, device="cuda", dtype=dt)
            G = torch.empty(B, p, device="cuda", dtype=dt)
            O = torch.empty(B, n, device="cuda", dtype=dt)
            t1 = bench(lambda: torch.mm(R, X, out=G))
            t2 = bench(lambda: torch.mm(W, X.T, out=O))
            fl = 2 * B * n * p
            print(json.dumps(dict(mode=mode, B=B, t_RX_ms=t1 * 1e3, t_WXt_ms=t2 * 1e3,
                                  tflops_RX=fl / t1 / 1e12, tflops_WXt=fl / t2 / 1e12,
                                  per_col_us=(t1 + t2) / B * 1e6)), flush=True)
            del R, W, G, O
        del X
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
