"""Learning rules for the classical network M1 (eqs. 1 and 5 of the proposal).

All weight matrices are float64, symmetric, with zero diagonal.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def hebbian(xi: np.ndarray) -> np.ndarray:
    """W_ij = (1/N) sum_mu xi_i^mu xi_j^mu, W_ii = 0.  xi: (P, N)."""
    P, N = xi.shape
    x = xi.astype(np.float64)
    W = x.T @ x / N
    np.fill_diagonal(W, 0.0)
    return W


@njit(cache=True)
def _storkey_step_nb(W, x):
    N = x.shape[0]
    h = W @ x
    dW = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                dW[i, j] = 0.0
                continue
            hij = h[i] - W[i, j] * x[j]          # sum_{k != i,j} W_ik x_k  (W_ii = 0)
            hji = h[j] - W[j, i] * x[i]
            dW[i, j] = (x[i] * x[j] - x[i] * hji - hij * x[j]) / N
    for i in range(N):
        for j in range(N):
            W[i, j] += dW[i, j]


def storkey_step(W: np.ndarray, x: np.ndarray) -> None:
    """In-place Storkey (1997) update with one pattern x (+-1, length N).

    W_ij += (1/N) [x_i x_j - x_i h_ji - h_ij x_j],
    h_ij = sum_{k != i, j} W_ik x_k.
    The diagonal is kept at zero.
    """
    _storkey_step_nb(W, x.astype(np.float64))


def storkey(xi: np.ndarray, checkpoints=None):
    """Storkey weights for patterns xi (P, N), stored in order.

    If ``checkpoints`` (sorted list of P values) is given, returns a dict
    {P: W after the first P patterns}; otherwise the final W.
    """
    P, N = xi.shape
    W = np.zeros((N, N))
    out = {}
    cps = set(checkpoints or [])
    for mu in range(P):
        storkey_step(W, xi[mu])
        if (mu + 1) in cps:
            out[mu + 1] = W.copy()
    if checkpoints is not None:
        return out
    return W


def hebbian_prefixes(xi: np.ndarray, checkpoints):
    """{P: Hebbian W for the first P patterns} computed incrementally."""
    P, N = xi.shape
    out = {}
    W = np.zeros((N, N))
    prev = 0
    for cp in sorted(checkpoints):
        blk = xi[prev:cp].astype(np.float64)
        W += blk.T @ blk / N
        prev = cp
        Wc = W.copy()
        np.fill_diagonal(Wc, 0.0)
        out[cp] = Wc
    return out


def stream_clipped(xi: np.ndarray, A: float) -> np.ndarray:
    """Parisi (1986) palimpsest: W <- clip(W + xi xi^T / N, -A/sqrt(N), A/sqrt(N)).

    Patterns are stored one at a time in row order; A = inf gives the
    unbounded Hebbian network.
    """
    P, N = xi.shape
    W = np.zeros((N, N))
    bound = A / np.sqrt(N)
    for mu in range(P):
        x = xi[mu].astype(np.float64)
        W += np.outer(x, x) / N
        if np.isfinite(bound):
            np.clip(W, -bound, bound, out=W)
    np.fill_diagonal(W, 0.0)
    return W


def stream_decay(xi: np.ndarray, lam: float) -> np.ndarray:
    """Mezard-Nadal-Toulouse palimpsest: W <- lam W + xi xi^T / N.

    Closed form: W = (1/N) sum_mu lam^(P-1-mu) xi^mu xi^mu^T.
    """
    P, N = xi.shape
    w = lam ** np.arange(P - 1, -1, -1, dtype=np.float64)
    x = xi.astype(np.float64)
    W = (x * w[:, None]).T @ x / N
    np.fill_diagonal(W, 0.0)
    return W
