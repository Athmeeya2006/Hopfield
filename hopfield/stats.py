"""Statistics (proposal section 5).

* The pattern set is the unit of replication; cues within a set share the
  same weights, so cue outcomes are first averaged within each set.
* Summaries: mean +- SD across sets and a 95% percentile bootstrap CI
  (resampling sets).
* Model comparisons are paired (same patterns and cues), via a paired
  bootstrap over sets.
* One pre-specified primary test per hypothesis, Holm corrected.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 2000
BOOT_SEED = 12345


def bootstrap_ci(values, stat=np.mean, n_boot=N_BOOT, alpha=0.05, seed=BOOT_SEED, axis=0):
    """Percentile CI of ``stat`` over resampled sets (rows of ``values``)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    n = v.shape[axis]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(np.take(v, ix, axis=axis), axis=axis) for ix in idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return lo, hi


def summarize(values, axis=0):
    """dict(mean, sd, lo, hi, n) across sets (rows)."""
    v = np.asarray(values, float)
    lo, hi = bootstrap_ci(v, np.nanmean, axis=axis)
    return dict(mean=np.nanmean(v, axis=axis), sd=np.nanstd(v, axis=axis, ddof=1),
                lo=lo, hi=hi, n=v.shape[axis])


def bootstrap_stat(per_set_arrays, fn, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap a statistic computed from several arrays that share the set
    axis 0 (paired resampling).  ``fn(*arrays) -> float``.
    Returns (estimate, lo, hi, boots)."""
    arrs = [np.asarray(a) for a in per_set_arrays]
    n = arrs[0].shape[0]
    rng = np.random.default_rng(seed)
    est = fn(*arrs)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, size=n)
        boots[b] = fn(*[a[ix] for a in arrs])
    lo = np.nanpercentile(boots, 2.5, method="lower")
    hi = np.nanpercentile(boots, 97.5, method="higher")
    return est, lo, hi, boots


def boot_p_value(boots, null=0.0):
    """Two-sided bootstrap p-value for H0: statistic = null
    (twice the smaller tail mass of the bootstrap distribution around null)."""
    b = np.asarray(boots, float)
    b = b[~np.isnan(b)]  # +-inf (censored) values are kept: they lie on one side of null
    if b.size == 0:
        return np.nan
    lo = np.mean(b <= null)
    hi = np.mean(b >= null)
    p = 2 * min(lo, hi)
    return float(min(1.0, max(p, 1.0 / b.size)))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict {name: p}."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    adj = {}
    running = 0.0
    for r, k in enumerate(names):
        a = min(1.0, (m - r) * pvals[k])
        running = max(running, a)
        adj[k] = running
    return adj
