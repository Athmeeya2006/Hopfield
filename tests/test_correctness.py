"""Correctness checks (proposal section 5, 'Correctness checks').

(i)   E never increases under asynchronous updates (all binary models, and
      Ramsauer's energy for the continuous model M3);
(ii)  stored patterns are fixed points at low load;
(iii) the Hebbian critical load approaches 0.138 -- checked by Part A
      (experiments/partA.py), with a fast smoke version here;
(iv)  M2 with n = 2 reproduces M1 exactly (update by update).
"""
import numpy as np
import pytest

from hopfield import kernels
from hopfield.learning import hebbian, storkey, storkey_step, stream_clipped, stream_decay
from hopfield.models import Classical, ContinuousExp, Dense, TwoLayer
from hopfield.patterns import cues_from, flip_bits, prototype_patterns, random_patterns, take_per_prototype
from hopfield.metrics import overlap, classify, TARGET, PROTO, crossing


def _models(xi, structured=False):
    return [
        Classical.hebbian(xi),
        Classical.storkey(xi),
        Dense(xi, 2), Dense(xi, 3), Dense(xi, 4), Dense(xi, "exp"),
        TwoLayer(xi, "power", n=2), TwoLayer(xi, "power", n=4),
        TwoLayer(xi, "softmax", beta=0.1), TwoLayer(xi, "softmax", beta=1.0),
    ]


# ------------------------------------------------------------------ (i)
@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5])
def test_energy_non_increasing_per_sweep(alpha):
    rng = np.random.default_rng(1)
    N = 80
    xi = random_patterns(rng, max(2, int(alpha * N)), N)
    for model in _models(xi):
        for c in range(5):
            cue = flip_bits(rng, xi[c % len(xi)], 0.3)
            r = model.recall(cue, seed=c, record=True)
            dE = np.diff(r.E_traj)
            assert np.all(dE <= 1e-9 * (1 + np.abs(r.E_traj[:-1]))), (model.name, r.E_traj)


def test_energy_non_increasing_random_states_and_clipped_weights():
    rng = np.random.default_rng(2)
    N = 60
    xi = random_patterns(rng, 40, N)
    for W in (stream_clipped(xi, 0.5), stream_decay(xi, 0.9), storkey(xi)):
        m = Classical(W)
        for c in range(5):
            s0 = random_patterns(rng, 1, N)[0]
            r = m.recall(s0, seed=c, record=True)
            assert np.all(np.diff(r.E_traj) <= 1e-12)
            # final state is a true fixed point
            h = W @ r.state
            assert np.all(h * r.state >= -1e-10)


def test_dense_update_takes_lower_energy_state():
    """Each neuron takes the sign that gives the lower energy (eq. 2)."""
    rng = np.random.default_rng(3)
    N = 30
    xi = random_patterns(rng, 25, N)
    for n in (2, 3, 4, "exp"):
        m = Dense(xi, n)
        for _ in range(50):
            s = random_patterns(rng, 1, N)[0].astype(np.float64)
            a = xi.astype(np.float64) @ s
            i = rng.integers(N)
            dE = kernels._dense_dE(m.XiT, a, s, i, m.mode, m.n, float(N))
            sp, sm = s.copy(), s.copy()
            sp[i], sm[i] = 1, -1
            Ep, Em = m.energy(sp), m.energy(sm)
            assert np.sign(dE) == np.sign(Em - Ep) or abs(Em - Ep) < 1e-9 * abs(Ep)


def test_continuous_energy_non_increasing():
    rng = np.random.default_rng(4)
    N = 64
    xi = random_patterns(rng, 30, N)
    for beta in (0.01, 0.05, 0.2, 1.0):
        m = ContinuousExp(xi, beta)
        for c in range(5):
            r = m.recall(flip_bits(rng, xi[c], 0.3), record=True)
            assert np.all(np.diff(r.E_traj) <= 1e-9)


def test_lesioned_twolayer_energy_non_increasing():
    rng = np.random.default_rng(5)
    N, P = 60, 30
    xi = random_patterns(rng, P, N)
    W = xi.T.astype(float) * (rng.random((N, P)) > 0.4) + 0.3 * rng.standard_normal((N, P))
    g = (rng.random(P) > 0.3).astype(float) * rng.random(P)
    for kind in ("power", "softmax"):
        m = TwoLayer(xi, kind, n=4, beta=0.5, g=g, W=W)
        for c in range(5):
            r = m.recall(random_patterns(rng, 1, N)[0], seed=c, record=True)
            assert np.all(np.diff(r.E_traj) <= 1e-9 * (1 + np.abs(r.E_traj[:-1])))


# ------------------------------------------------------------------ (ii)
def test_stored_patterns_are_fixed_points_at_low_load():
    rng = np.random.default_rng(6)
    N = 500
    xi = random_patterns(rng, 10, N)  # alpha = 0.02
    for model in _models(xi):
        for mu in range(len(xi)):
            r = model.recall(xi[mu], seed=mu)
            assert r.time == 0 and np.array_equal(r.state, xi[mu]), model.name
    m3 = ContinuousExp(xi, beta=0.1)
    for mu in range(len(xi)):
        r = m3.recall(xi[mu])
        assert overlap(r.state, xi[mu]) > 0.999


# ------------------------------------------------------------------ (iii)
def test_hebbian_capacity_smoke():
    N = 1000
    res = {}
    for alpha in (0.08, 0.20):
        ok = []
        for s in range(3):
            rng = np.random.default_rng(100 + s)
            xi = random_patterns(rng, int(alpha * N), N)
            m = Classical.hebbian(xi)
            for c in range(5):
                r = m.recall(flip_bits(rng, xi[c], 0.05), seed=c)
                ok.append(overlap(r.state, xi[c]) >= 0.95)
        res[alpha] = np.mean(ok)
    assert res[0.08] > 0.95 and res[0.20] < 0.2


# ------------------------------------------------------------------ (iv)
@pytest.mark.parametrize("T", [0.0, 0.4])
def test_dense_n2_equals_classical(T):
    rng = np.random.default_rng(7)
    N = 100
    for P in (5, 14, 25):
        xi = random_patterns(rng, P, N)
        m1, m2 = Classical.hebbian(xi), Dense(xi, 2)
        for c in range(10):
            cue = flip_bits(rng, xi[c % P], 0.2)
            r1 = m1.recall(cue, seed=c, T=T, max_sweeps=20, quench_sweeps=10 if T else 0,
                           refs=xi, record=True)
            r2 = m2.recall(cue, seed=c, T=T, max_sweeps=20, quench_sweeps=10 if T else 0,
                           refs=xi, record=True)
            assert np.array_equal(r1.state, r2.state)
            assert r1.sweeps == r2.sweeps and r1.time == r2.time
            np.testing.assert_allclose(r1.m_traj, r2.m_traj)
            # energies equal up to the constant -P/2 (diagonal term)
            np.testing.assert_allclose(r1.E_traj - P / 2, r2.E_traj, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------ misc
def test_storkey_matches_reference_formula():
    rng = np.random.default_rng(8)
    N, P = 12, 5
    xi = random_patterns(rng, P, N).astype(float)
    W = np.zeros((N, N))
    for mu in range(P):
        x = xi[mu]
        Wn = W.copy()
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                hij = sum(W[i, k] * x[k] for k in range(N) if k not in (i, j))
                hji = sum(W[j, k] * x[k] for k in range(N) if k not in (i, j))
                Wn[i, j] = W[i, j] + (x[i] * x[j] - x[i] * hji - hij * x[j]) / N
        W = Wn
    np.testing.assert_allclose(storkey(xi.astype(np.int8)), W, atol=1e-12)


def test_prototype_statistics():
    rng = np.random.default_rng(9)
    N = 4000
    for c in (0.2, 0.6):
        protos, mems = prototype_patterns(rng, 2, 6, N, c)
        m_proto = np.mean([overlap(mems[k, j], protos[k]) for k in range(2) for j in range(6)])
        m_sib = np.mean([overlap(mems[0, a], mems[0, b]) for a in range(6) for b in range(a + 1, 6)])
        assert abs(m_proto - c) < 0.03 and abs(m_sib - c * c) < 0.03
    x = random_patterns(rng, 1, 200)[0]
    assert np.isclose(overlap(flip_bits(rng, x, 0.1), x), 0.8)


def test_classify_and_crossing():
    rng = np.random.default_rng(10)
    xi = random_patterns(rng, 3, 100)
    proto = random_patterns(rng, 1, 100)[0]
    assert classify(xi[1], 1, xi, proto)[0] == TARGET
    assert classify(proto, 1, xi, proto)[0] == PROTO
    assert np.isclose(crossing([0, 1, 2, 3], [1, 0.8, 0.4, 0.0]), 1.75)
