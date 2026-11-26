"""Shared infrastructure for the experiment scripts."""
from __future__ import annotations

import json
import os
import sys

# one BLAS thread per worker process (parallelism comes from the process pool)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Replication (proposal section 5): >= 50 independent pattern sets, ~20 cues per set.
QUICK = os.environ.get("QUICK", "0") == "1"   # smoke-test mode
N_SETS = 4 if QUICK else 50
N_CUES = 5 if QUICK else 20
WORKERS = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) - 1)))


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=WORKERS)
    return _POOL


def pmap(fn, tasks, chunksize=1):
    """Parallel map preserving order (one persistent worker pool per script)."""
    tasks = list(tasks)
    if WORKERS <= 1 or len(tasks) == 1:
        return [fn(t) for t in tasks]
    return list(_pool().map(fn, tasks, chunksize=chunksize))


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None if np.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.floating):
        return _clean(float(o))
    return o


def save_json(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=1, cls=_Enc)
    return path


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save_npz(name, **arrays):
    path = os.path.join(RESULTS, name)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(name):
    return dict(np.load(os.path.join(RESULTS, name), allow_pickle=False))


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[{self.label}] start", flush=True)
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] done in {time.time() - self.t:.1f}s", flush=True)


def warmup():
    """Compile/load all Numba kernels once in the parent process (cached)."""
    from hopfield.models import Classical, Dense, TwoLayer
    from hopfield.patterns import random_patterns
    xi = random_patterns(np.random.default_rng(0), 3, 16)
    Classical.hebbian(xi).recall(xi[0], 0, record=True, refs=xi)
    Classical.hebbian(xi).recall(xi[0], 0, T=0.1, quench_sweeps=2)
    Dense(xi, 3).recall(xi[0], 0, record=True, refs=xi)
    Dense(xi, "exp").recall(xi[0], 0, T=0.1, quench_sweeps=2)
    TwoLayer(xi, "softmax").recall(xi[0], 0, record=True, refs=xi)
    TwoLayer(xi, "power").recall(xi[0], 0)
