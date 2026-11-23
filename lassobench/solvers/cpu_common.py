"""Shared helpers for CPU per-target baselines (owned by bench-harness)."""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp


def n_cpus():
    return len(os.sched_getaffinity(0))


def per_target_parallel(fn, T, n_jobs=None, backend="threading"):
    """Run fn(t) for t in range(T) with joblib, BLAS pinned to 1 thread per worker."""
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits
    n_jobs = n_jobs or n_cpus()
    with threadpool_limits(1):
        return Parallel(n_jobs=min(n_jobs, T), backend=backend)(delayed(fn)(t) for t in range(T))


def _batch_worker(fn_factory, X_path, Y, alphas, ts):
    from threadpoolctl import threadpool_limits
    # copy-on-write memmap: writeable (celer's cython wants a non-const buffer), pages shared across workers
    X = np.load(X_path, mmap_mode="c")
    with threadpool_limits(1):
        fn = fn_factory(X, Y, alphas)
        return [fn(t) for t in ts]


def _tmpdir(nbytes):
    for d in (os.environ.get("LASSOBENCH_TMP"), "/dev/shm", os.environ.get("TMPDIR"), "/tmp"):
        if d and os.path.isdir(d):
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize > 1.5 * nbytes:
                return d
    raise OSError("no temp dir with enough space for X")


def per_target_processes(fn_factory, X, Y, alphas, n_jobs=None):
    """Process-based variant: targets split into n_jobs batches; X shared via one F-ordered .npy memmap.

    fn_factory(X, Y, alphas) -> fn(t); must be a picklable module-level callable.
    """
    import tempfile
    from joblib import Parallel, delayed
    T = Y.shape[1]
    n_jobs = min(n_jobs or n_cpus(), T)
    batches = [b for b in np.array_split(np.arange(T), n_jobs) if b.size]
    fd, path = tempfile.mkstemp(suffix=".npy", dir=_tmpdir(X.nbytes))
    os.close(fd)
    try:
        np.save(path, np.asfortranarray(X))
        outs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_batch_worker)(fn_factory, path, Y, alphas, b) for b in batches)
    finally:
        os.remove(path)
    res = [None] * T
    for b, o in zip(batches, outs):
        for t, r in zip(b, o):
            res[t] = r
    return res


def assemble(cols, p, A):
    """cols[t] = (p, A) sparse/dense path for target t -> list of A CSC (p, T)."""
    T = len(cols)
    C = [sp.csc_matrix(c) for c in cols]
    out = []
    for a in range(A):
        out.append(sp.hstack([c[:, a] for c in C]).tocsc() if T else sp.csc_matrix((p, 0)))
    return out
