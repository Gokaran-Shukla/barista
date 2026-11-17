"""Single-node parallelism.

All independent work — the global fit of every mechanism, the cross-validation refits,
the two-channel screening/refinement and the series-resistance test — is distributed
over a process pool.  The pool size defaults to the CPUs this process may use
(``os.sched_getaffinity``), which honours SLURM/PBS/LSF cpusets; override with
``BARISTA_WORKERS`` or ``--workers``.  Each worker pins BLAS/OpenMP to one thread so
that N workers use exactly N cores.

Only plain data (mechanism keys, parameter dicts, arrays) cross process boundaries;
mechanism objects are rebuilt inside the workers.
"""
from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def available_cpus() -> int:
    env = os.environ.get("BARISTA_WORKERS")
    if env:
        return max(1, int(env))
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NUM_PPN", "NSLOTS", "LSB_DJOB_NUMPROC"):
        v = os.environ.get(var)
        if v and v.isdigit():
            n = int(v)
            break
    else:
        n = None
    try:
        aff = len(os.sched_getaffinity(0))
    except AttributeError:
        aff = os.cpu_count() or 1
    return max(1, min(n, aff) if n else aff)


def _init_worker():
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore")


class Pool:
    """Thin wrapper around ProcessPoolExecutor that degrades to serial execution."""

    def __init__(self, workers=None):
        self.workers = available_cpus() if workers in (None, 0) else max(1, int(workers))
        self._ex = None

    def __enter__(self):
        if self.workers > 1:
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
            self._ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_init_worker)
        return self

    def __exit__(self, *exc):
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)

    def map(self, func, tasks, cost=None, progress=None, label=None):
        """Run func(*task) for every task; results in input order.  Costly tasks are submitted first."""
        tasks = list(tasks)
        if not tasks:
            return []
        order = list(range(len(tasks)))
        if cost is not None:
            order.sort(key=lambda i: -cost(tasks[i]))
        if self._ex is None:
            out = [None] * len(tasks)
            for i in order:
                if progress and label:
                    progress(label(tasks[i]))
                out[i] = func(*tasks[i])
            return out
        futs = {}
        for i in order:
            futs[i] = self._ex.submit(func, *tasks[i])
        out = [None] * len(tasks)
        from concurrent.futures import as_completed
        rev = {f: i for i, f in futs.items()}
        for f in as_completed(futs.values()):
            i = rev[f]
            out[i] = f.result()
            if progress and label:
                progress("done: " + label(tasks[i]))
        return out


# ---------------------------------------------------------------------------
# worker tasks (top-level functions so that they pickle under 'spawn' too)
# ---------------------------------------------------------------------------
def task_fit(key, data, dev, hints, n_starts):
    from .models import REGISTRY
    from .fitting import fit
    return fit(REGISTRY[key], data, dev, hints=hints, n_starts=n_starts)


def task_cv(key, data, dev, res, cov):
    from .models import REGISTRY
    from .fitting import cross_validate
    return cross_validate(REGISTRY[key], data, dev, res, cov)


def task_combo(ka, kb, pa, pb, data, dev, screen, hints, n_starts, seed):
    from .models import REGISTRY
    from .fitting import fit, Parallel
    pm = Parallel(REGISTRY[ka], REGISTRY[kb], pa, pb)
    return fit(pm, data, dev, hints=hints, n_starts=n_starts, seed=seed, screen=screen)


def task_series(key, params, area, data, dev):
    from .models import REGISTRY
    from .fitting import fit, WithSeriesR
    sm = WithSeriesR(REGISTRY[key], params, area)
    return fit(sm, data, dev, n_starts=3)


COST = {"coherent": 40, "tat": 25, "sclc_trap": 15, "sclc_exp": 2, "vrh": 2}


def cost_of(key):
    return sum(COST.get(k, 1) for k in str(key).split("+"))
