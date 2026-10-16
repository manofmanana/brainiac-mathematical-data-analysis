"""Activation functions and their exact derivatives.

Each activation is a pair ``(f, f')`` acting element-wise on NumPy arrays.
Derivatives are written out analytically (no autograd) and are verified against
central finite differences in ``tests/test_activations.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

Array = np.ndarray

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _sigmoid(x: Array) -> Array:
    # Numerically stable: never exponentiates a large positive number.
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _gelu(x: Array) -> Array:
    # Exact GELU: x * Phi(x), with Phi the standard normal CDF.
    return 0.5 * x * (1.0 + erf(x / _SQRT2))


def _gelu_deriv(x: Array) -> Array:
    # d/dx [x * Phi(x)] = Phi(x) + x * phi(x)
    cdf = 0.5 * (1.0 + erf(x / _SQRT2))
    pdf = _INV_SQRT_2PI * np.exp(-0.5 * x**2)
    return cdf + x * pdf


@dataclass(frozen=True)
class ActivationFn:
    name: str
    f: Callable[[Array], Array]
    df: Callable[[Array], Array]
    formula: str  # LaTeX, shown in the app


ACTIVATIONS: dict[str, ActivationFn] = {
    "relu": ActivationFn("relu", lambda x: np.maximum(0.0, x), lambda x: (x > 0).astype(float), r"\max(0, x)"),
    "leaky_relu": ActivationFn(
        "leaky_relu",
        lambda x: np.where(x > 0, x, 0.01 * x),
        lambda x: np.where(x > 0, 1.0, 0.01),
        r"\max(0.01x,\, x)",
    ),
    "sigmoid": ActivationFn(
        "sigmoid", _sigmoid, lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)), r"\sigma(x) = \frac{1}{1 + e^{-x}}"
    ),
    "tanh": ActivationFn("tanh", np.tanh, lambda x: 1.0 - np.tanh(x) ** 2, r"\tanh(x)"),
    "gelu": ActivationFn("gelu", _gelu, _gelu_deriv, r"x\,\Phi(x)"),
    "linear": ActivationFn("linear", lambda x: x, lambda x: np.ones_like(x, dtype=float), r"x"),
}


def get_activation(name: str) -> ActivationFn:
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"Unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}") from None
