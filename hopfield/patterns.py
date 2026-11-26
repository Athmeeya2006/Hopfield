"""Synthetic memories and cues (proposal section 3.1).

* Random memories: xi_i^mu in {-1,+1} with equal probability.
* Prototype-structured memories: K prototypes zeta^k in {-1,+1}^N that are
  never stored.  Each memory copies one prototype and flips each bit
  independently with probability (1-c)/2, so its expected overlap with the
  prototype is c and with a sibling is c**2.  The prototype is the unstudied
  "lure" of the DRM paradigm.
* Cues: flip a fraction eta of the bits of a stored memory (initial overlap
  1-2*eta), or present a corrupted prototype.
"""
from __future__ import annotations

import numpy as np


def random_patterns(rng: np.random.Generator, P: int, N: int) -> np.ndarray:
    """P x N array of +-1 (int8)."""
    return (2 * rng.integers(0, 2, size=(P, N), dtype=np.int8) - 1).astype(np.int8)


def prototype_patterns(rng: np.random.Generator, K: int, p: int, N: int, c: float):
    """Prototype-structured memories.

    Returns
    -------
    protos : (K, N) int8, the unstored prototypes
    mems   : (K, p, N) int8, mems[k, j] is the j-th memory of prototype k
    """
    protos = random_patterns(rng, K, N)
    flip = rng.random(size=(K, p, N)) < (1.0 - c) / 2.0
    mems = np.where(flip, -protos[:, None, :], protos[:, None, :]).astype(np.int8)
    return protos, mems


def take_per_prototype(mems: np.ndarray, p: int):
    """First p memories of every prototype, flattened to (K*p, N).

    Also returns the prototype label of every row.  Because the first p
    memories are a prefix of the first p' > p memories, sweeps over p use
    nested pattern sets (paired across loads as well as across models).
    """
    K = mems.shape[0]
    sub = mems[:, :p, :].reshape(K * p, -1)
    labels = np.repeat(np.arange(K), p)
    return np.ascontiguousarray(sub), labels


def flip_bits(rng: np.random.Generator, x: np.ndarray, eta: float) -> np.ndarray:
    """Copy of +-1 vector x with exactly round(eta*N) bits flipped."""
    N = x.shape[-1]
    k = int(round(eta * N))
    y = x.copy()
    if k > 0:
        idx = rng.choice(N, size=k, replace=False)
        y[idx] = -y[idx]
    return y


def cues_from(rng: np.random.Generator, sources: np.ndarray, eta: float) -> np.ndarray:
    """One corrupted cue per row of ``sources``."""
    return np.stack([flip_bits(rng, s, eta) for s in sources]).astype(np.int8)


# ---------------------------------------------------------------- demo letters
_LETTERS = {
    "A": [
        "......########......",
        "......########......",
        ".....##########.....",
        ".....##########.....",
        "....############....",
        "....############....",
        "...##############...",
        "...##############...",
        "...######..######...",
        "..#######..#######..",
        "..################..",
        "..################..",
        ".##################.",
        ".##################.",
        ".#######....#######.",
        ".######......######.",
        "#######......#######",
        "######........######",
        "####............####",
        "....................",
    ],
    "H": [
        "....................",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".##################.",
        ".##################.",
        ".##################.",
        ".##################.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        ".#####........#####.",
        "....................",
        "....................",
    ],
    "X": [
        "....................",
        "..#####......#####..",
        "..#####......#####..",
        "...#####....#####...",
        "....#####..#####....",
        "....############....",
        ".....##########.....",
        ".....##########.....",
        "......########......",
        "......########......",
        ".....##########.....",
        ".....##########.....",
        "....############....",
        "....#####..#####....",
        "...#####....#####...",
        "...####......####...",
        "..#####......#####..",
        ".######......######.",
        "....................",
        "....................",
    ],
}


def letter(ch: str) -> np.ndarray:
    """20x20 letter bitmap as a +-1 vector of length 400 (used by the demo)."""
    rows = _LETTERS[ch]
    arr = np.array([[1 if c == "#" else -1 for c in r] for r in rows], dtype=np.int8)
    return arr.ravel()


LETTERS = tuple(_LETTERS)
