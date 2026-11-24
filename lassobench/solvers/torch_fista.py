"""Batched FISTA Lasso on GPU (gpu-prox session).

All (target, alpha[, fold]) problems share one X on the device. The solver runs a pool of
"slots"; each slot is one (target, fold) walking down a contiguous segment of the alpha path with
warm starts. When a slot's problem reaches the duality-gap tolerance its sparse solution is pulled
to the host and the slot moves to the next alpha (momentum reset). Slots that finish their segment
are refilled from a pending queue or compacted away, so the batch stays dense on the GPU.

  segments=1      pure warm-start path: T slots live, each walking all A alphas.
  segments=A      all alphas simultaneously from a cold (zero) start: T*A slots.
  segments=s      in between.

Storage is slot-major: W is (B, p) so the live batch is always a contiguous prefix W[:B].
Per iteration: G = R_z @ X (B,n)x(n,p) and Xw = W @ X^T (B,p)x(p,n); elementwise prox fused per row-chunk.

Options (constructor kwargs): dtype ('float32'|'float64'|'bfloat16'), tf32 (bool), segments, restart
(gradient restart, O'Donoghue & Candes 2015), check_every (gap check period, iterations), mem_frac,
max_batch, screen ('none'|'strong': sequential strong rule + KKT check on full p), profile (sync timers).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch

from lassobench.api import PathResult, Solver
from lassobench.solvers._torch_common import (DTYPES, PhaseTimer, assemble_csc, batched_gap,
                                               default_device, device_name, extract_rows,
                                               free_gpu_bytes, fused_prox_, fused_prox_available,
                                               lipschitz, tf32_mode, to_device)


class TorchFISTA(Solver):
    name = "torch_fista"
    device = "cuda"

    def __init__(self, tol=1e-4, max_iter=10000, dtype="float32", tf32=False, segments=1, restart=True,
                 check_every=10, mem_frac=0.85, max_batch=None, profile=False, verbose=False,
                 fused=True, screen=False, screen_max_frac=0.5, cuda_graph=False, tol_margin=0.5, device=None,
                 **kw):
        super().__init__(tol=tol, max_iter=max_iter, **kw)
        self.dtype = dtype
        self.tf32 = tf32
        self.segments = segments
        self.restart = restart
        self.check_every = check_every
        self.mem_frac = mem_frac
        self.max_batch = max_batch
        self.profile = profile
        self.verbose = verbose
        self.fused = fused
        self.screen = screen
        self.cuda_graph = cuda_graph
        self.tol_margin = tol_margin   # internal stop at tol_margin * tol (fp32 gap + fp32 coef storage)
        self.screen_max_frac = screen_max_frac
        self.dev = torch.device(device) if device else default_device()

    # ------------------------------------------------------------------ public API
    def config(self) -> dict:
        return dict(tol=self.tol, max_iter=self.max_iter, dtype=self.dtype, tf32=self.tf32,
                    segments=self.segments, restart=self.restart, check_every=self.check_every,
                    fused=self.fused, screen=self.screen, cuda_graph=self.cuda_graph,
                    tol_margin=self.tol_margin)

    def fit_path(self, X, Y, alphas) -> PathResult:
        return self.fit_folds(X, Y, alphas, train_masks=None)[0]

    def fit_cv(self, X, Y, alphas, folds):
        """Harness CV protocol (lassobench.runner): all K folds (and, with cv_refit='batched', the
        full-data refit) solved as ONE batched run on the full X with 0/1 row masks per fold.

        cv_refit: 'batched' (default; refit path for all alphas inside the same batch) or 'after'
        (second run on alphas[:max chosen + 1] after selection, like the default protocol).
        """
        import scipy.sparse as sp
        X = np.asarray(X); Y = np.asarray(Y)
        n = X.shape[0]
        A, T = alphas.shape
        K = len(folds)
        mode = self.kw.get("cv_refit", "batched")
        masks = np.zeros((K + (mode == "batched"), n), bool)
        for k, (tr, _) in enumerate(folds):
            masks[k, tr] = True
        if mode == "batched":
            masks[K] = True
        t0 = time.perf_counter()
        res = self.fit_folds(X, Y, alphas, masks)
        t_batch = time.perf_counter() - t0
        mse = np.zeros((K, A, T))
        for k, (_, va) in enumerate(folds):
            Xva = X[va].astype(np.float64); Yva = Y[va].astype(np.float64)
            for a in range(A):
                pred = np.asarray((res[k].coefs[a].T @ Xva.T).T)
                mse[k, a] = ((Yva - pred) ** 2).mean(0)
        cv_mse = mse.mean(0)
        best = np.argmin(cv_mse, axis=0)
        info = dict(time_batched_s=t_batch, cv_refit=mode,
                    fold_infos=[{k: v for k, v in r.info.items() if not isinstance(v, np.ndarray)} for r in res])
        if mode == "batched":
            final = res[K]
        else:
            amax = int(best.max())
            t0 = time.perf_counter()
            final = self.fit_path(X, Y, np.ascontiguousarray(alphas[: amax + 1]))
            info["refit_time_s"] = time.perf_counter() - t0
        cols = [final.coefs[best[t]][:, t] for t in range(T)]
        return dict(coef=sp.hstack(cols).tocsc(), best_idx=best, cv_mse=cv_mse, info=info)

    def fit_folds(self, X, Y, alphas, train_masks=None) -> list[PathResult]:
        """Solve the path for several row subsets of X at once (CV folds) in one batched run.

        train_masks: None (one fit on all rows) or bool array (F, n); fold f uses rows train_masks[f].
        All folds share the alpha grid `alphas` (A, T). Returns one PathResult per fold.
        """
        t_start = time.perf_counter()
        X = np.asarray(X); Y = np.asarray(Y)
        alphas = np.asarray(alphas, dtype=np.float64)
        n, p = X.shape
        T = Y.shape[1]
        A = alphas.shape[0]
        dev = self.dev
        xdt = DTYPES[self.dtype]
        timer = PhaseTimer(dev, self.profile)
        with tf32_mode(self.tf32), torch.no_grad():
            t0 = time.perf_counter()
            Xg = to_device(X, dev, xdt)
            ydt = torch.float64 if xdt == torch.float64 else torch.float32
            Yg = to_device(Y, dev, ydt)
            if train_masks is None:
                masks = None
                F = 1
            else:
                masks = torch.as_tensor(np.asarray(train_masks), device=dev).to(ydt)
                F = masks.shape[0]
            sync_ = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
            sync_()
            t_transfer = time.perf_counter() - t0
            t0 = time.perf_counter()
            if masks is None:
                L_f = np.array([lipschitz(Xg)])
                n_f = np.array([float(n)])
            else:
                L_f = np.array([lipschitz(Xg, masks[f]) for f in range(F)])
                n_f = masks.sum(1).double().cpu().numpy()
            sync_()
            t_lip = time.perf_counter() - t0

            segs = np.array_split(np.arange(A), max(1, min(self.segments, A)))
            slots = [(t, f, int(s[0]), int(s[-1]) + 1) for f in range(F) for s in segs for t in range(T)]
            t0 = time.perf_counter()
            pieces, iters, gaps, stats = self._engine(Xg, Yg, masks, n_f, L_f, alphas, slots, timer, F, A, T)
            sync_()
            t_solve = time.perf_counter() - t0
            del Xg, Yg
        t0 = time.perf_counter()
        results = []
        for f in range(F):
            coefs = [assemble_csc(pieces.get((f, a), {}), p, T) for a in range(A)]
            info = dict(solver=self.name, config=self.config(), gpu=device_name(dev),
                        iters=iters[f], rel_gap=gaps[f], lipschitz=float(L_f[f]),
                        time_transfer_s=t_transfer, time_lipschitz_s=t_lip, time_engine_s=t_solve,
                        **stats)
            results.append(PathResult(coefs=coefs, info=info))
        t_asm = time.perf_counter() - t0
        for r in results:
            r.info["time_assemble_s"] = t_asm
            r.info["time_fit_s"] = time.perf_counter() - t_start
            if self.profile:
                r.info["phase_s"] = dict(timer.t)
        return results

    # ------------------------------------------------------------------ engine
    def _batch_capacity(self, p, n, esize, n_slots):
        if self.max_batch:
            return int(min(self.max_batch, n_slots))
        free = free_gpu_bytes(self.dev) * self.mem_frac
        # persistent per slot: W, Wp, G (p each) + bf16 cast temp, plus ~6 (n,) rows
        per = p * (3 * esize + (2 if self.dtype == "bfloat16" else 0)) + 8 * n * 8
        # row-chunk prox temps are bounded separately (~3 * chunk_elems)
        cap = int(max(1, (free - 3 * 6.4e7 * esize) // per))
        return int(max(1, min(cap, n_slots)))

    def _engine(self, Xg, Yg, masks, n_f, L_f, alphas, slots, timer, F, A, T):
        dev = Xg.device
        n, p = Xg.shape
        xdt = Xg.dtype
        wdt = torch.float64 if xdt == torch.float64 else torch.float32
        esize = torch.finfo(wdt).bits // 8
        Bcap = self._batch_capacity(p, n, esize, len(slots))
        alphas_g = torch.as_tensor(alphas, device=dev, dtype=torch.float64)
        # Active feature set (screening). S is None -> all p features, Xa is Xg itself.
        # With screening, W/Wp/G hold only the q = |S| active columns and Xa = X[:, S] (gathered copy).
        S = None
        S_host = None
        Xa = Xg
        q = p

        def mm_RX(R, out, Xm=None):        # out (B,q) = R (B,n) @ Xa (n,q)
            Xm = Xa if Xm is None else Xm
            if xdt == torch.bfloat16:
                out.copy_(R.to(xdt) @ Xm)
            else:
                torch.mm(R.to(xdt), Xm, out=out)

        def mm_WXt(Wb, out):      # out (B,n) = W (B,q) @ Xa^T (q,n)
            if xdt == torch.bfloat16:
                out.copy_(Wb.to(xdt) @ Xa.T)
            else:
                torch.mm(Wb, Xa.T, out=out)

        # persistent buffers (slot-major)
        W = torch.zeros(Bcap, p, device=dev, dtype=wdt)
        Wp = torch.zeros(Bcap, p, device=dev, dtype=wdt)
        G = torch.zeros(Bcap, p, device=dev, dtype=wdt)
        Xw = torch.zeros(Bcap, n, device=dev, dtype=wdt)
        Xwp = torch.zeros(Bcap, n, device=dev, dtype=wdt)
        Ysel = torch.zeros(Bcap, n, device=dev, dtype=wdt)
        M = torch.ones(Bcap, n, device=dev, dtype=wdt) if masks is not None else None
        tk = torch.ones(Bcap, device=dev, dtype=torch.float64)
        alpha = torch.zeros(Bcap, device=dev, dtype=torch.float64)
        ninv = torch.zeros(Bcap, device=dev, dtype=torch.float64)
        Lb = torch.ones(Bcap, device=dev, dtype=torch.float64)
        ynorm2 = torch.zeros(Bcap, device=dev, dtype=torch.float64)
        # host bookkeeping
        s_t = np.zeros(Bcap, np.int64); s_f = np.zeros(Bcap, np.int64)
        s_a = np.zeros(Bcap, np.int64); s_a1 = np.zeros(Bcap, np.int64)
        s_it = np.zeros(Bcap, np.int64); alive = np.zeros(Bcap, bool)
        pending = deque(slots)
        pieces: dict = {}
        iters = np.zeros((F, A, T), np.int64)
        gaps = np.full((F, A, T), np.nan)
        Yt = Yg.T.contiguous()  # (T, n)
        n_f_g = torch.as_tensor(n_f, device=dev, dtype=torch.float64)
        L_f_g = torch.as_tensor(L_f, device=dev, dtype=torch.float64)

        def load(pos, items):
            """Install pending slots into positions `pos` with zero (cold) start."""
            if not len(pos):
                return
            pos_g = torch.as_tensor(pos, device=dev)
            t = np.array([it[0] for it in items]); f = np.array([it[1] for it in items])
            a0 = np.array([it[2] for it in items]); a1 = np.array([it[3] for it in items])
            s_t[pos], s_f[pos], s_a[pos], s_a1[pos], s_it[pos] = t, f, a0, a1, 0
            alive[pos] = True
            tg = torch.as_tensor(t, device=dev); fg = torch.as_tensor(f, device=dev)
            W[pos_g] = 0; Wp[pos_g] = 0; Xw[pos_g] = 0; Xwp[pos_g] = 0
            ys = Yt[tg].to(wdt)
            if M is not None:
                mk = masks[fg].to(wdt)
                M[pos_g] = mk
                ys = ys * mk
            Ysel[pos_g] = ys
            ynorm2[pos_g] = (ys.double() ** 2).sum(1)
            tk[pos_g] = 1.0
            alpha[pos_g] = alphas_g[torch.as_tensor(a0, device=dev), tg]
            ninv[pos_g] = 1.0 / n_f_g[fg]
            Lb[pos_g] = L_f_g[fg]

        # initial fill
        B = min(Bcap, len(pending))
        load(np.arange(B), [pending.popleft() for _ in range(B)])

        chunk_rows = max(1, int(6.4e7 // p))
        use_fused = self.fused and fused_prox_available(W)
        s_restart = torch.zeros(Bcap, device=dev, dtype=torch.float32 if use_fused else torch.float64)
        total_it = 0
        n_restarts = 0
        n_checks = 0
        max_batch_seen = B
        slot_iters = 0  # sum over iterations of live batch size (work measure)
        guard = int(self.max_iter) * A * max(1, len(slots)) + 10
        n_rebuilds = 0
        q_trace = []
        c_rows = max(1, int(1.6e8 // p))       # slot rows per full-gradient chunk when screening
        alpha_prev_of = lambda a_idx, t_idx: alphas_g[torch.clamp(a_idx - 1, min=0), t_idx]  # noqa: E731

        def fista_iter(W, Wp, G, Xw, Xwp, B):
            """One FISTA step on the live prefix [:B]. Returns the rotated buffer tuple."""
            with timer("momentum"):
                tkB = tk[:B]
                t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * tkB * tkB))
                beta = ((tkB - 1.0) / t_next).to(wdt)[:, None]
                tk[:B] = t_next
                if not use_fused:                                    # fused kernel forms z on the fly
                    Zb = Wp[:B]
                    Zb.mul_(-beta).addcmul_(W[:B], 1.0 + beta)      # z = (1+b) w - b w_prev
                Xz = Xwp[:B]
                Xz.mul_(-beta).addcmul_(Xw[:B], 1.0 + beta)
                Rz = Xz - Ysel[:B]
                if M is not None:
                    Rz.mul_(M[:B])
            with timer("mm_grad"):
                mm_RX(Rz, G[:B])                                     # G = X^T (Xz - y), slot-major
            with timer("prox"):
                step = (ninv[:B] / Lb[:B]).to(wdt)
                thr = (alpha[:B] / Lb[:B]).to(wdt)
                if use_fused:
                    fused_prox_(G[:B], Wp[:B], W[:B], beta[:, 0].contiguous(), step, thr,
                                s_restart[:B], restart=self.restart)
                else:
                    step, thr = step[:, None], thr[:, None]
                    for i in range(0, B, chunk_rows):
                        j = min(B, i + chunk_rows)
                        g, z, w = G[i:j], Wp[i:j], W[i:j]
                        th = thr[i:j]
                        g.mul_(-step[i:j]).add_(z)                   # u = z - grad / L
                        g.sub_(torch.clamp(g, -th, th))              # soft threshold
                        if self.restart:
                            s_restart[i:j] = ((z - g) * (g - w)).sum(1, dtype=s_restart.dtype)
            with timer("mm_fwd"):
                mm_WXt(G[:B], Xwp[:B])                               # X w_new (overwrites Xz)
            if self.restart:
                rs = s_restart[:B] > 0
                tk[:B] = torch.where(rs, torch.ones_like(tk[:B]), tk[:B])
            # rotate: (W, Wp, G) <- (Wnew, W, Zold); (Xw, Xwp) <- (Xwnew, Xw)
            return G, W, Wp, Xwp, Xw

        # CUDA graphs: one captured graph = GR iterations (GR=6 returns the 3-way/2-way buffer rotation to
        # identity), replayed between gap checks. Recaptured whenever B, q or a buffer changes.
        GR = 6
        use_graph = bool(self.cuda_graph) and dev.type == "cuda" and not self.profile
        check_every = int(self.check_every)
        if use_graph:
            check_every = max(GR, GR * round(check_every / GR))
        graph, gkey = None, None
        n_captures = 0

        while B > 0 and total_it < guard:
            # ---------------- check_every FISTA steps on W[:B]
            if use_graph and n_checks >= 1:
                key = (B, q, W.data_ptr(), Wp.data_ptr(), G.data_ptr(), Xw.data_ptr(), Xwp.data_ptr(),
                       Xa.data_ptr())
                reps = check_every // GR
                if key != gkey:
                    graph = None
                    torch.cuda.synchronize(dev)
                    side = torch.cuda.Stream(dev)
                    side.wait_stream(torch.cuda.current_stream(dev))
                    with torch.cuda.stream(side):                     # eager warm-up block (real iterations)
                        bufs = (W, Wp, G, Xw, Xwp)
                        for _ in range(GR):
                            bufs = fista_iter(*bufs, B)
                    torch.cuda.current_stream(dev).wait_stream(side)
                    assert bufs[0] is W and bufs[3] is Xw
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        bufs = (W, Wp, G, Xw, Xwp)
                        for _ in range(GR):
                            bufs = fista_iter(*bufs, B)
                    gkey = key
                    n_captures += 1
                    reps -= 1
                for _ in range(reps):
                    graph.replay()
                n_do = check_every
            else:
                n_do = check_every - (total_it % check_every)
                bufs = (W, Wp, G, Xw, Xwp)
                for _ in range(n_do):
                    bufs = fista_iter(*bufs, B)
                W, Wp, G, Xw, Xwp = bufs
            s_it[:B] += n_do
            total_it += n_do
            slot_iters += B * n_do

            # ---------------- duality gap check on W[:B]
            n_checks += 1
            with timer("gap"):
                R = Ysel[:B] - Xw[:B]
                if M is not None:
                    R.mul_(M[:B])
                l1 = torch.linalg.vector_norm(W[:B], ord=1, dim=1).double()
                if not self.screen:
                    mm_RX(R, G[:B])                                  # X^T r (G is free here)
                    Cinf = torch.linalg.vector_norm(G[:B], ord=float("inf"), dim=1).double()
                    gap, rel = batched_gap(R, Cinf, l1, Ysel[:B], alpha[:B], ninv[:B], ynorm2[:B])
                else:
                    # full-p gradient pass in slot chunks: honest gap over all p + strong-rule set for the
                    # alpha each slot will solve next (its next grid alpha if it converges now)
                    Cinf = torch.empty(B, device=dev, dtype=torch.float64)
                    gap = torch.empty_like(Cinf); rel = torch.empty_like(Cinf)
                    keep = torch.zeros(p, device=dev, dtype=torch.bool)
                    a_idx = torch.as_tensor(s_a[:B], device=dev); t_idx = torch.as_tensor(s_t[:B], device=dev)
                    a_end = torch.as_tensor(s_a1[:B], device=dev)
                    for i in range(0, B, c_rows):
                        j = min(B, i + c_rows)
                        Cc = R[i:j].to(xdt) @ Xg
                        Cinf[i:j] = torch.linalg.vector_norm(Cc, ord=float("inf"), dim=1).double()
                        gap[i:j], rel[i:j] = batched_gap(R[i:j], Cinf[i:j], l1[i:j], Ysel[i:j], alpha[i:j],
                                                         ninv[i:j], ynorm2[i:j])
                        cv = (rel[i:j] <= self.tol * self.tol_margin) | (gap[i:j] <= 1e-30)
                        ai, ti = a_idx[i:j], t_idx[i:j]
                        nxt_ok = cv & (ai + 1 < a_end[i:j])
                        a_new = torch.where(nxt_ok, alphas_g[torch.clamp(ai + 1, max=A - 1), ti], alpha[i:j])
                        a_old = torch.where(nxt_ok, alpha[i:j], alpha_prev_of(ai, ti))
                        thr_c = ((2 * a_new - a_old) / ninv[i:j]).to(Cc.dtype)
                        thr_c = torch.where(cv & ~nxt_ok, torch.full_like(thr_c, float("inf")), thr_c)
                        keep |= (Cc.abs() >= thr_c[:, None]).any(0)
                        del Cc
                    # never drop current support (or momentum support): keeps Xw/Xwp consistent
                    if S is None:
                        keep |= (W[:B] != 0).any(0) | (Wp[:B] != 0).any(0)
                    else:
                        keep[S] |= (W[:B] != 0).any(0) | (Wp[:B] != 0).any(0)
                if self.restart:
                    n_restarts += int((s_restart[:B] > 0).sum())
                conv = ((rel <= self.tol * self.tol_margin) | (gap <= 1e-30)).cpu().numpy()
                relh = rel.cpu().numpy()
            conv |= s_it[:B] >= self.max_iter
            if not conv.any() and not self.screen:
                continue
            with timer("extract"):
                # advance converged slots along their alpha segment; re-check with the same residual
                todo = np.nonzero(conv)[0]
                while len(todo):
                    for pos, (r, v) in zip(todo, extract_rows(W, torch.as_tensor(todo, device=dev))):
                        if S_host is not None:
                            r = S_host[r].astype(np.int32)
                        f, a, t = s_f[pos], s_a[pos], s_t[pos]
                        pieces.setdefault((f, a), {})[t] = (r, v)
                        iters[f, a, t] = s_it[pos]
                        gaps[f, a, t] = relh[pos]
                    s_a[todo] += 1
                    s_it[todo] = 0
                    nxt = todo[s_a[todo] < s_a1[todo]]
                    alive[todo[s_a[todo] >= s_a1[todo]]] = False
                    if not len(nxt):
                        break
                    ng = torch.as_tensor(nxt, device=dev)
                    alpha[ng] = alphas_g[torch.as_tensor(s_a[nxt], device=dev),
                                         torch.as_tensor(s_t[nxt], device=dev)]
                    tk[ng] = 1.0
                    gap2, rel2 = batched_gap(R[ng], Cinf[ng], l1[ng], Ysel[ng], alpha[ng], ninv[ng], ynorm2[ng])
                    c2 = ((rel2 <= self.tol * self.tol_margin) | (gap2 <= 1e-30)).cpu().numpy()
                    relh[nxt] = rel2.cpu().numpy()
                    todo = nxt[c2]
            with timer("bookkeep"):
                dead = np.nonzero(~alive[:B])[0]
                # refill dead positions from the pending queue (cold start)
                if len(dead) and pending:
                    k = min(len(dead), len(pending))
                    load(dead[:k], [pending.popleft() for _ in range(k)])
                    dead = dead[k:]
                # grow into free capacity
                if not len(dead) and pending and B < Bcap:
                    k = min(Bcap - B, len(pending))
                    load(np.arange(B, B + k), [pending.popleft() for _ in range(k)])
                    B += k
                # compact: move alive tail slots into dead holes, shrink B
                if len(dead):
                    newB = B - len(dead)
                    holes = dead[dead < newB]
                    movers = np.nonzero(alive[newB:B])[0] + newB
                    assert len(holes) == len(movers)
                    if len(holes):
                        hg = torch.as_tensor(holes, device=dev); mg = torch.as_tensor(movers, device=dev)
                        for buf in [W, Wp, Xw, Xwp, Ysel, tk, alpha, ninv, Lb, ynorm2] + ([M] if M is not None else []):
                            buf[hg] = buf[mg]
                        for arr in (s_t, s_f, s_a, s_a1, s_it, alive):
                            arr[holes] = arr[movers]
                    alive[newB:B] = False
                    B = newB
                max_batch_seen = max(max_batch_seen, B)
            if self.screen:
                with timer("screen"):
                    in_S = torch.zeros(p, device=dev, dtype=torch.bool)
                    if S is None:
                        in_S[:] = True
                    else:
                        in_S[S] = True
                    n_keep = int(keep.sum())
                    need = bool((keep & ~in_S).any())
                    if need or n_keep < 0.5 * q:
                        if n_keep > self.screen_max_frac * p:
                            S_new = None
                        else:
                            S_new = torch.nonzero(keep | (torch.arange(p, device=dev) == 0))[:, 0]
                        if not (S_new is None and S is None):
                            n_rebuilds += 1
                            q_new = p if S_new is None else int(S_new.numel())
                            # map old compact columns into the new compact space
                            pos_new = torch.full((p,), -1, device=dev, dtype=torch.long)
                            pos_new[S_new if S_new is not None else torch.arange(p, device=dev)] = \
                                torch.arange(q_new, device=dev)
                            old_full = S if S is not None else torch.arange(p, device=dev)
                            dst = pos_new[old_full]
                            okm = dst >= 0
                            src_cols = torch.nonzero(okm)[:, 0]
                            dst_cols = dst[okm]
                            newbufs = []
                            for buf in (W, Wp):
                                nb = torch.zeros(Bcap, q_new, device=dev, dtype=wdt)
                                nb[:, dst_cols] = buf[:, src_cols]
                                newbufs.append(nb)
                            W, Wp = newbufs
                            del newbufs
                            G = torch.zeros(Bcap, q_new, device=dev, dtype=wdt)
                            S = S_new
                            S_host = None if S is None else S.cpu().numpy()
                            Xa = Xg if S is None else Xg.index_select(1, S)
                            q = q_new
                    q_trace.append(q)
            if self.verbose:
                print(f"it {total_it} B {B} pending {len(pending)} median rel {np.median(relh):.2e}", flush=True)

        stats = dict(total_iters=total_it, slot_iters=int(slot_iters), batch_capacity=int(Bcap),
                     n_checks=n_checks, n_restarts=n_restarts,
                     n_unconverged=int(np.sum(gaps > self.tol * self.tol_margin)), n_screen_rebuilds=n_rebuilds,
                     n_graph_captures=n_captures, check_every_eff=check_every,
                     q_mean=float(np.mean(q_trace)) if q_trace else float(p),
                     q_max=int(max(q_trace)) if q_trace else p)
        return pieces, iters, gaps, stats


SOLVERS = {
    "torch_fista": TorchFISTA,
}
