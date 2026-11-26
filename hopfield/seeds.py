"""Deterministic seeding.

Every random object in the project is generated from one master seed through
``numpy.random.SeedSequence``.  A seed is addressed by a tuple of integers
(experiment code, condition indices, pattern-set index, ...), so any single
condition or pattern set can be regenerated on its own, independently of the
order in which conditions are run.

The *pattern set* is the unit of replication.  Its patterns, cues and the
random update orders used by the dynamics are drawn from separate child
streams, so all models see identical patterns and cues (paired design).
"""
from __future__ import annotations

import zlib

import numpy as np

MASTER_SEED = 20261017  # one master seed for the whole project


def _code(name: str) -> int:
    """Stable integer code for a string label (crc32, platform independent)."""
    return zlib.crc32(name.encode("utf8")) & 0x7FFFFFFF


def seed_seq(*key) -> np.random.SeedSequence:
    """SeedSequence addressed by ``key`` (ints or strings)."""
    spawn_key = tuple(_code(k) if isinstance(k, str) else int(k) for k in key)
    return np.random.SeedSequence(entropy=MASTER_SEED, spawn_key=spawn_key)


def rng(*key) -> np.random.Generator:
    return np.random.default_rng(seed_seq(*key))


def int_seed(*key) -> int:
    """A 31-bit integer seed (for the Numba kernels' internal generator)."""
    return int(seed_seq(*key).generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)
