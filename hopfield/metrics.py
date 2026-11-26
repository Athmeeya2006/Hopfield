"""Metrics and final-state classification (proposal section 5)."""
from __future__ import annotations

import numpy as np

THRESH = 0.95                   # P_ret = Pr[m_final >= 0.95]
ALT_THRESH = (0.90, 0.99)       # robustness checks of the threshold

# final-state categories
TARGET, OTHER, PROTO, SPURIOUS = 0, 1, 2, 3
CATEGORIES = ("target", "other memory", "prototype", "spurious")


def overlap(a, b):
    """m = (1/N) sum_i a_i b_i  (a, b may be 1-D or 2-D, last axis = N)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a @ b.T / a.shape[-1]


def binarize(x):
    """sign with ties -> +1 (only needed for the continuous model M3)."""
    return np.where(np.asarray(x) >= 0, 1, -1).astype(np.int8)


def classify(state, target_idx, memories, prototype=None, thresh=THRESH):
    """Classify one final state.

    Categories, checked in this order:
      target       m(state, cued memory)            >= thresh
      other memory max m(state, other stored memory) >= thresh
      prototype    m(sign(state), own prototype)     >= thresh
      spurious     anything else (mixtures, blends, random minima)

    Overlaps with stored memories use the raw state (so a continuous blend of
    memories is *not* counted as one memory); the prototype test uses the
    binarised state.  For binary models both are the same.
    ``target_idx`` may be None (e.g. a lure cue has no target).
    Returns (category, m_target, m_other_max, m_proto).
    """
    m = overlap(state, memories)
    if target_idx is None:
        m_t = np.nan
        m_o = m.max() if m.size else -1.0
    else:
        m_t = m[target_idx]
        mo = np.delete(m, target_idx)
        m_o = mo.max() if mo.size else -1.0
    m_p = overlap(binarize(state), prototype) if prototype is not None else np.nan
    if target_idx is not None and m_t >= thresh:
        cat = TARGET
    elif m_o >= thresh:
        cat = OTHER
    elif prototype is not None and m_p >= thresh:
        cat = PROTO
    else:
        cat = SPURIOUS
    return cat, m_t, m_o, m_p


def participation_ratio(p):
    """Effective number of memories in a softmax weight vector, 1/sum p^2."""
    p = np.asarray(p)
    return 1.0 / np.sum(p ** 2, axis=0)


def crossing(x, y, level=0.5, direction="down"):
    """First x where y crosses ``level`` (linear interpolation).

    direction='down': y goes from >= level to < level (largest x with
    y >= level, interpolated).  'up': y goes from < level to >= level.
    Returns nan if there is no crossing within the grid; -inf / +inf if y is
    entirely on one side (censored below / above the grid).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if direction == "down":
        ok = y >= level
        if not ok.any():
            return -np.inf
        if ok.all():
            return np.inf
        # last index that is >= level before the first failure run ends
        i = np.where(ok)[0]
        # use the first downward crossing after the largest x still >= level
        j = i[-1]
        if j == len(x) - 1:
            return np.inf
        x0, x1, y0, y1 = x[j], x[j + 1], y[j], y[j + 1]
    else:
        ok = y >= level
        if ok.all():
            return -np.inf
        if not ok.any():
            return np.inf
        j = np.where(ok)[0][0]
        if j == 0:
            return -np.inf
        x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
    if y1 == y0:
        return x0
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def log_crossing(x, y, level=0.5, direction="down"):
    """Like ``crossing`` but interpolates in log x (for geometric grids)."""
    r = crossing(np.log(np.asarray(x, float)), y, level, direction)
    return np.exp(r) if np.isfinite(r) else r
