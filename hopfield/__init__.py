"""Classical and modern Hopfield networks: a from-scratch simulator.

INCM 2026, Project 15 -- "From Memorization to Generalization".
"""
from . import kernels, learning, metrics, models, patterns, seeds, stats  # noqa: F401
from .models import Classical, ContinuousExp, Dense, Recall, TwoLayer  # noqa: F401
