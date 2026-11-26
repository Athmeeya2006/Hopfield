"""The four model families of the proposal (section 3.2).

M1  Classical     -- ``Classical(W)`` with Hebbian or Storkey W (eq. 1)
M2  Dense         -- ``Dense(xi, n)`` with F(x)=x^n, or ``Dense(xi, 'exp')`` (eq. 2)
M3  Continuous    -- ``ContinuousExp(xi, beta)``: x <- X softmax(beta X^T x) (eq. 3)
M4  Two-layer     -- ``TwoLayer(xi, ...)`` feature <-> memory neurons

Every model exposes ``recall(cue, seed, ...) -> Recall``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels
from .learning import hebbian, storkey

MAX_SWEEPS = 50  # "a run stops at a fixed point or after about 50 sweeps"
_NOREF = np.zeros((0, 1))


@dataclass
class Recall:
    state: np.ndarray        # final state (int8 +-1 for binary models; float for M3)
    sweeps: int              # sweeps (iterations) performed in total
    noise_sweeps: int        # sweeps performed at temperature T (before quench)
    time: float              # recall time: last flip / N, in sweeps (M3: iterations)
    converged: bool          # reached a fixed point within the limit
    m_traj: np.ndarray       # (sweeps+1, R) overlaps with refs, if recorded
    E_traj: np.ndarray       # (sweeps+1,) energy, if recorded
    weights: np.ndarray | None = None  # M3 only: final softmax weights over memories


def _refs(refs, N):
    if refs is None:
        return np.zeros((0, N))
    return np.ascontiguousarray(np.atleast_2d(refs), dtype=np.float64)


class Classical:
    """M1. E = -1/2 sum_{i != j} W_ij s_i s_j."""

    name = "M1"

    def __init__(self, W: np.ndarray):
        self.W = np.ascontiguousarray(W, dtype=np.float64)
        self.N = W.shape[0]

    @classmethod
    def hebbian(cls, xi):
        return cls(hebbian(xi))

    @classmethod
    def storkey(cls, xi):
        return cls(storkey(xi))

    def recall(self, cue, seed, T=0.0, max_sweeps=MAX_SWEEPS, quench_sweeps=0,
               refs=None, record=False) -> Recall:
        out = kernels.run_classical(self.W, np.ascontiguousarray(cue, dtype=np.int8),
                                    float(T), int(max_sweeps), int(quench_sweeps),
                                    int(seed), _refs(refs, self.N), bool(record))
        return Recall(*out)

    def energy(self, s):
        s = s.astype(np.float64)
        return -0.5 * s @ self.W @ s


class Dense:
    """M2. E = -sum_mu F(xi^mu . s) with F(x) = x^n or e^x."""

    def __init__(self, xi: np.ndarray, n):
        self.xi = np.ascontiguousarray(xi, dtype=np.int8)
        self.XiT = np.ascontiguousarray(self.xi.T)
        self.P, self.N = xi.shape
        if n == "exp":
            self.mode, self.n = 1, 0
        else:
            self.mode, self.n = 0, int(n)
        self.name = "M2-exp" if self.mode else f"M2-n{self.n}"

    def recall(self, cue, seed, T=0.0, max_sweeps=MAX_SWEEPS, quench_sweeps=0,
               refs=None, record=False) -> Recall:
        out = kernels.run_dense(self.XiT, np.ascontiguousarray(cue, dtype=np.int8),
                                self.mode, self.n, float(T), int(max_sweeps),
                                int(quench_sweeps), int(seed), _refs(refs, self.N),
                                bool(record))
        return Recall(*out)

    def energy(self, s):
        a = self.xi.astype(np.float64) @ s.astype(np.float64)
        if self.mode == 0:
            return -np.sum(a ** self.n) / (2.0 * self.N ** (self.n - 1))
        mx = a.max()
        return -(mx + np.log(np.exp(a - mx).sum()))


def _softmax_cols(Z):
    Z = Z - Z.max(axis=0, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=0, keepdims=True)


class ContinuousExp:
    """M3 (Ramsauer et al. 2021). x_new = X softmax(beta X^T x).

    Memories are the columns of X (+-1 entries).  The update is synchronous
    and deterministic; a run stops when max|dx| < tol or after 50 iterations.
    Energy: E = -(1/beta) log sum_mu exp(beta xi^mu . x) + x.x/2 (+ const),
    which the update never increases.
    """

    name = "M3"

    def __init__(self, xi: np.ndarray, beta: float, values: np.ndarray | None = None):
        self.X = np.ascontiguousarray(xi.T, dtype=np.float64)  # N x P
        self.N, self.P = self.X.shape
        self.beta = float(beta)

    def energy(self, x):
        z = self.beta * (self.X.T @ x)
        mx = z.max()
        return -(mx + np.log(np.exp(z - mx).sum())) / self.beta + 0.5 * x @ x

    def recall(self, cue, seed=None, max_iter=MAX_SWEEPS, tol=1e-6, refs=None,
               record=False, **_) -> Recall:
        x = np.asarray(cue, dtype=np.float64).copy()
        R = None if refs is None else _refs(refs, self.N)
        ms, Es = [], []
        if record:
            ms.append(R @ x / self.N if R is not None else np.zeros(0))
            Es.append(self.energy(x))
        it = 0
        conv = False
        last_change = 0
        p = None
        for it in range(1, max_iter + 1):
            z = self.beta * (self.X.T @ x)
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            xn = self.X @ p
            dx = np.max(np.abs(xn - x))
            x = xn
            if record:
                ms.append(R @ x / self.N if R is not None else np.zeros(0))
                Es.append(self.energy(x))
            if dx < tol:
                conv = True
                break
            last_change = it
        return Recall(x, it, it, float(last_change), conv,
                      np.array(ms) if record else np.zeros((0, 0)),
                      np.array(Es) if record else np.zeros(0), p)

    def recall_batch(self, cues, max_iter=MAX_SWEEPS, tol=1e-6):
        """Vectorised recall of many cues (C x N).  Returns (states C x N,
        iterations-to-convergence per cue, softmax weights P x C)."""
        Xc = np.asarray(cues, dtype=np.float64).T.copy()  # N x C
        C = Xc.shape[1]
        iters = np.full(C, max_iter, dtype=float)
        done = np.zeros(C, bool)
        P_ = None
        for it in range(1, max_iter + 1):
            P_ = _softmax_cols(self.beta * (self.X.T @ Xc))
            Xn = self.X @ P_
            dx = np.max(np.abs(Xn - Xc), axis=0)
            Xc = np.where(done[None, :], Xc, Xn)
            newly = (~done) & (dx < tol)
            iters[newly] = it - 1  # last iteration that changed the state
            done |= newly
            if done.all():
                break
        P_ = _softmax_cols(self.beta * (self.X.T @ Xc))
        return Xc.T, iters, P_


class TwoLayer:
    """M4 (Krotov & Hopfield 2021): feature neurons <-> memory neurons.

    Parameters
    ----------
    xi    : (P, N) memories; the synapses are W_i,mu = xi_i^mu (pairwise only)
    kind  : 'power' (F(h) = h^n, n even) or 'softmax' (inverse temperature beta)
    g     : optional strength of each memory neuron (0 = lesioned neuron)
    W     : optional explicit synapse matrix (N x P), e.g. lesioned or noisy
    """

    def __init__(self, xi, kind="softmax", n=4, beta=1.0, g=None, W=None):
        xi = np.asarray(xi)
        self.P, self.N = xi.shape
        self.W = np.ascontiguousarray(xi.T if W is None else W, dtype=np.float64)
        self.g = np.ones(self.P) if g is None else np.ascontiguousarray(g, dtype=np.float64)
        self.mode = 0 if kind == "power" else 1
        self.n = int(n)
        self.beta = float(beta)
        self.name = f"M4-{kind}"

    def recall(self, cue, seed, max_sweeps=MAX_SWEEPS, refs=None, record=False, **_) -> Recall:
        out = kernels.run_twolayer(self.W, self.g, self.mode, self.n, self.beta,
                                   np.ascontiguousarray(cue, dtype=np.int8),
                                   int(max_sweeps), int(seed), _refs(refs, self.N),
                                   bool(record))
        return Recall(*out)
