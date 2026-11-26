"""Numba kernels for the binary asynchronous dynamics (M1, M2, M4).

Conventions shared by all kernels
---------------------------------
* Asynchronous updates in a fresh random order every sweep; one sweep = N
  single-neuron updates.  Random orders and Glauber coin flips use Numba's
  internal generator seeded with ``seed`` (deterministic, and identical across
  models for the same seed, so M2 with n = 2 reproduces M1 update by update).
* Deterministic update (T = 0): s_i <- sign of the energy-lowering direction;
  an exact tie keeps the current state (this makes E non-increasing).
* Glauber dynamics (T > 0), eq. (4): P(s_i = +1) = 1 / (1 + exp(-dE_i / T)),
  where dE_i = E(s_i = -1) - E(s_i = +1).  For M1 dE_i = 2 h_i.
* A run stops at a fixed point (a full sweep without a flip) or after
  ``max_sweeps`` sweeps.  With T > 0, ``quench_sweeps`` > 0 adds a final
  zero-temperature descent to the nearest fixed point.
* ``refs`` (R x N) are reference patterns whose overlaps m = refs . s / N are
  recorded after every sweep together with the energy, if ``record``.
* Returned recall time = (index of the last update that flipped a neuron)/N,
  i.e. convergence time in (fractional) sweeps.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

TIE_EPS = 1e-10


@njit(cache=True)
def _overlaps(refs, s, out_row):
    R, N = refs.shape
    for r in range(R):
        acc = 0.0
        for j in range(N):
            acc += refs[r, j] * s[j]
        out_row[r] = acc / N


# --------------------------------------------------------------------- M1
@njit(cache=True)
def _classical_energy(h, s):
    e = 0.0
    for i in range(s.shape[0]):
        e += s[i] * h[i]
    return -0.5 * e


@njit(cache=True)
def run_classical(W, s0, T, max_sweeps, quench_sweeps, seed, refs, record):
    """M1: E = -1/2 sum_ij W_ij s_i s_j with symmetric, zero-diagonal W."""
    N = s0.shape[0]
    R = refs.shape[0]
    np.random.seed(seed)
    s = s0.astype(np.float64).copy()
    h = W @ s
    L = max_sweeps + quench_sweeps + 1 if record else 1
    traj_m = np.full((L, R), np.nan)
    traj_E = np.full(L, np.nan)
    if record:
        _overlaps(refs, s, traj_m[0])
        traj_E[0] = _classical_energy(h, s)
    t = 0
    last_flip = 0
    k = 0  # sweeps performed in total
    sweeps_noise = 0
    converged = False
    for phase in range(2):
        temp = T if phase == 0 else 0.0
        limit = max_sweeps if phase == 0 else (quench_sweeps if T > 0.0 else 0)
        if phase == 1 and converged and T == 0.0:
            break
        for sw in range(limit):
            order = np.random.permutation(N)
            flips = 0
            for kk in range(N):
                i = order[kk]
                hi = h[i]
                if temp > 0.0:
                    x = -2.0 * hi / temp
                    if x > 700.0:
                        p = 0.0
                    else:
                        p = 1.0 / (1.0 + math.exp(x))
                    new = 1.0 if np.random.random() < p else -1.0
                else:
                    if hi > TIE_EPS:
                        new = 1.0
                    elif hi < -TIE_EPS:
                        new = -1.0
                    else:
                        new = s[i]
                t += 1
                if new != s[i]:
                    s[i] = new
                    d = 2.0 * new
                    for j in range(N):
                        h[j] += d * W[i, j]
                    flips += 1
                    last_flip = t
            k += 1
            if phase == 0:
                sweeps_noise += 1
            if record:
                _overlaps(refs, s, traj_m[k])
                traj_E[k] = _classical_energy(h, s)
            if flips == 0:
                converged = True
                break
            else:
                converged = False
    return (s.astype(np.int8), k, sweeps_noise, last_flip / N, converged,
            traj_m[: k + 1], traj_E[: k + 1])


# --------------------------------------------------------------------- M2
@njit(cache=True)
def _powi(x, n):
    r = 1.0
    for _ in range(n):
        r *= x
    return r


@njit(cache=True)
def _dense_energy(a, mode, n, N):
    P = a.shape[0]
    if mode == 0:
        e = 0.0
        for mu in range(P):
            e += _powi(a[mu], n)
        return -e / (2.0 * _powi(N, n - 1))
    mx = -1e300
    for mu in range(P):
        if a[mu] > mx:
            mx = a[mu]
    acc = 0.0
    for mu in range(P):
        acc += math.exp(a[mu] - mx)
    return -(mx + math.log(acc))


@njit(cache=True)
def _dense_dE(XiT, a, s, i, mode, n, N):
    """dE_i = E(s_i=-1) - E(s_i=+1) (scaled), see module docstring."""
    P = a.shape[0]
    si = s[i]
    if mode == 0:
        D = 0.0
        for mu in range(P):
            x = XiT[i, mu]
            b = a[mu] - x * si
            D += _powi(b + x, n) - _powi(b - x, n)
        return D / (2.0 * _powi(N, n - 1))
    # exponential: work with the log-sum-exp energy (overflow free)
    mx = -1e300
    for mu in range(P):
        b = a[mu] - XiT[i, mu] * si
        if b > mx:
            mx = b
    sp = 0.0
    sm = 0.0
    for mu in range(P):
        x = XiT[i, mu]
        b = a[mu] - x * si - mx
        sp += math.exp(b + x)
        sm += math.exp(b - x)
    return math.log(sp) - math.log(sm)


@njit(cache=True)
def run_dense(XiT, s0, mode, n, T, max_sweeps, quench_sweeps, seed, refs, record):
    """M2: E = -sum_mu F(xi^mu . s), F(x) = x^n (mode 0) or e^x (mode 1).

    XiT is the (N, P) int8 transpose of the pattern matrix.  Each neuron takes
    the state that gives the lower energy (eq. 2).  For Glauber dynamics the
    polynomial energy is scaled by 1/(2 N^(n-1)) (n = 2 then equals M1
    exactly, up to a constant) and the exponential model uses the log-sum-exp
    energy -log sum_mu exp(xi^mu . s), a monotone transform of E.
    """
    N, P = XiT.shape
    R = refs.shape[0]
    np.random.seed(seed)
    s = s0.astype(np.float64).copy()
    a = np.zeros(P)
    for mu in range(P):
        acc = 0.0
        for j in range(N):
            acc += XiT[j, mu] * s[j]
        a[mu] = acc
    L = max_sweeps + quench_sweeps + 1 if record else 1
    traj_m = np.full((L, R), np.nan)
    traj_E = np.full(L, np.nan)
    if record:
        _overlaps(refs, s, traj_m[0])
        traj_E[0] = _dense_energy(a, mode, n, N)
    t = 0
    last_flip = 0
    k = 0
    sweeps_noise = 0
    converged = False
    for phase in range(2):
        temp = T if phase == 0 else 0.0
        limit = max_sweeps if phase == 0 else (quench_sweeps if T > 0.0 else 0)
        for sw in range(limit):
            order = np.random.permutation(N)
            flips = 0
            for kk in range(N):
                i = order[kk]
                dE = _dense_dE(XiT, a, s, i, mode, n, N)
                if temp > 0.0:
                    x = -dE / temp
                    if x > 700.0:
                        p = 0.0
                    else:
                        p = 1.0 / (1.0 + math.exp(x))
                    new = 1.0 if np.random.random() < p else -1.0
                else:
                    if dE > 0.0:
                        new = 1.0
                    elif dE < 0.0:
                        new = -1.0
                    else:
                        new = s[i]
                t += 1
                if new != s[i]:
                    s[i] = new
                    d = 2.0 * new
                    for mu in range(P):
                        a[mu] += d * XiT[i, mu]
                    flips += 1
                    last_flip = t
            k += 1
            if phase == 0:
                sweeps_noise += 1
            if record:
                _overlaps(refs, s, traj_m[k])
                traj_E[k] = _dense_energy(a, mode, n, N)
            if flips == 0:
                converged = True
                break
            else:
                converged = False
    return (s.astype(np.int8), k, sweeps_noise, last_flip / N, converged,
            traj_m[: k + 1], traj_E[: k + 1])


# --------------------------------------------------------------------- M4
@njit(cache=True)
def _twolayer_energy(hm, g, mode, n, beta, N):
    P = hm.shape[0]
    if mode == 0:
        e = 0.0
        for mu in range(P):
            e += g[mu] * _powi(hm[mu] / N, n)
        return -e * N
    mx = -1e300
    for mu in range(P):
        if g[mu] > 0.0:
            v = beta * hm[mu] + math.log(g[mu])
            if v > mx:
                mx = v
    if mx == -1e300:
        return 0.0
    acc = 0.0
    for mu in range(P):
        if g[mu] > 0.0:
            acc += math.exp(beta * hm[mu] + math.log(g[mu]) - mx)
    return -(mx + math.log(acc)) / beta


@njit(cache=True)
def run_twolayer(W, g, mode, n, beta, s0, max_sweeps, seed, refs, record):
    """M4: feature neurons v (binary) <-> memory neurons (fast, continuous).

    Memory neuron mu receives h_mu = sum_i W_i,mu v_i through pairwise
    synapses W (N x P) and has strength g_mu >= 0 (g_mu = 0: lesioned).
    Feature neuron i is updated asynchronously:
        v_i <- sign( sum_mu W_i,mu f_mu ),
    f_mu = g_mu F'(h_mu) with F(h) = h^n (mode 0, n even), or
    f_mu = softmax_mu(beta h_mu + log g_mu) (mode 1).  The Lyapunov energy is
    E = -sum_mu g_mu F(h_mu)  or  E = -(1/beta) log sum_mu g_mu exp(beta h_mu);
    convexity of F guarantees that E never increases.
    """
    N, P = W.shape
    R = refs.shape[0]
    np.random.seed(seed)
    s = s0.astype(np.float64).copy()
    hm = np.zeros(P)
    for mu in range(P):
        acc = 0.0
        for j in range(N):
            acc += W[j, mu] * s[j]
        hm[mu] = acc
    logg = np.empty(P)
    for mu in range(P):
        logg[mu] = math.log(g[mu]) if g[mu] > 0.0 else -np.inf
    L = max_sweeps + 1 if record else 1
    traj_m = np.full((L, R), np.nan)
    traj_E = np.full(L, np.nan)
    if record:
        _overlaps(refs, s, traj_m[0])
        traj_E[0] = _twolayer_energy(hm, g, mode, n, beta, N)
    t = 0
    last_flip = 0
    k = 0
    converged = False
    f = np.zeros(P)
    for sw in range(max_sweeps):
        order = np.random.permutation(N)
        flips = 0
        for kk in range(N):
            i = order[kk]
            # memory-neuron outputs
            if mode == 0:
                for mu in range(P):
                    f[mu] = g[mu] * _powi(hm[mu] / N, n - 1)
            else:
                mx = -1e300
                for mu in range(P):
                    if g[mu] > 0.0:
                        v = beta * hm[mu] + logg[mu]
                        if v > mx:
                            mx = v
                for mu in range(P):
                    if g[mu] > 0.0:
                        f[mu] = math.exp(beta * hm[mu] + logg[mu] - mx)
                    else:
                        f[mu] = 0.0
            field = 0.0
            for mu in range(P):
                field += W[i, mu] * f[mu]
            if field > 0.0:
                new = 1.0
            elif field < 0.0:
                new = -1.0
            else:
                new = s[i]
            t += 1
            if new != s[i]:
                s[i] = new
                d = 2.0 * new
                for mu in range(P):
                    hm[mu] += d * W[i, mu]
                flips += 1
                last_flip = t
        k += 1
        if record:
            _overlaps(refs, s, traj_m[k])
            traj_E[k] = _twolayer_energy(hm, g, mode, n, beta, N)
        if flips == 0:
            converged = True
            break
    return (s.astype(np.int8), k, k, last_flip / N, converged,
            traj_m[: k + 1], traj_E[: k + 1])
