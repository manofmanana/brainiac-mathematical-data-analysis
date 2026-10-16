"""Weight initialization schemes.

All schemes draw ``W`` with shape ``(fan_out, fan_in)`` and zero mean; they
differ in the variance, chosen so activations neither explode nor vanish as
signals pass through many layers:

=========  ==================================  ==========================
scheme     Var(W)                              designed for
=========  ==================================  ==========================
he         2 / fan_in                          ReLU-family
xavier     2 / (fan_in + fan_out)              tanh / sigmoid
lecun      1 / fan_in                          SELU / linear-ish
normal     0.05**2 (fixed)                     baseline: ignores fan-in
uniform    U(-sqrt(6/(in+out)), +...)          Glorot uniform
=========  ==================================  ==========================
"""

from __future__ import annotations

import numpy as np


def target_std(scheme: str, fan_in: int, fan_out: int) -> float:
    """Standard deviation each scheme aims for."""
    if scheme == "he":
        return float(np.sqrt(2.0 / fan_in))
    if scheme in ("xavier", "glorot"):
        return float(np.sqrt(2.0 / (fan_in + fan_out)))
    if scheme == "lecun":
        return float(np.sqrt(1.0 / fan_in))
    if scheme == "normal":
        return 0.05
    if scheme == "uniform":
        return float(np.sqrt(2.0 / (fan_in + fan_out)))  # Var(U(-a, a)) = a^2 / 3
    raise ValueError(f"Unknown init scheme {scheme!r}")


INIT_SCHEMES = ("he", "xavier", "lecun", "normal", "uniform")


def init_weights(fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator) -> np.ndarray:
    std = target_std(scheme, fan_in, fan_out)
    if scheme == "uniform":
        limit = np.sqrt(3.0) * std
        return rng.uniform(-limit, limit, size=(fan_out, fan_in))
    return rng.standard_normal((fan_out, fan_in)) * std
