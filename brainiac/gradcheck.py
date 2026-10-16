r"""Verify backpropagation against central finite differences.

For every parameter :math:`\theta_i`,

.. math::
    \frac{\partial L}{\partial \theta_i} \approx
    \frac{L(\theta + \varepsilon e_i) - L(\theta - \varepsilon e_i)}{2\varepsilon},

with :math:`O(\varepsilon^2)` truncation error. We report the relative error
:math:`\lVert g_{\text{analytic}} - g_{\text{numeric}} \rVert / (\lVert g_{\text{analytic}} \rVert + \lVert g_{\text{numeric}} \rVert)`;
values below ~1e-6 mean the analytic gradient is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import NeuralNetwork


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6, indices: np.ndarray | None = None) -> np.ndarray:
    """Central-difference gradient of scalar ``f`` at flat vector ``x`` (optionally only at ``indices``)."""
    x = np.array(x, dtype=float)
    grad = np.full_like(x, np.nan)
    for i in range(x.size) if indices is None else indices:
        orig = x[i]
        x[i] = orig + eps
        f_plus = f(x)
        x[i] = orig - eps
        f_minus = f(x)
        x[i] = orig
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12))


@dataclass
class GradCheckResult:
    overall: float
    per_layer: list[tuple[str, float]]  # (parameter name, relative error)
    analytic: np.ndarray
    numeric: np.ndarray
    checked: np.ndarray  # indices compared

    @property
    def passed(self) -> bool:
        return self.overall < 1e-5


def gradient_check(
    net: NeuralNetwork, X: np.ndarray, Y: np.ndarray, eps: float = 1e-6, max_params: int = 400, seed: int = 0
) -> GradCheckResult:
    """Compare backprop with finite differences on (a random subset of) the parameters."""
    theta = net.get_flat_params()
    analytic = net.flat_grad(theta, X, Y)
    rng = np.random.default_rng(seed)
    idx = (
        np.arange(theta.size)
        if theta.size <= max_params
        else np.sort(rng.choice(theta.size, max_params, replace=False))
    )
    numeric = numerical_gradient(lambda t: net.flat_loss(t, X, Y), theta, eps, idx)

    per_layer = []
    start = 0
    for li, layer in enumerate(net.dense_layers, start=1):
        for pname, p in (("W", layer.W), ("b", layer.b)):
            mask = (idx >= start) & (idx < start + p.size)
            if mask.any():
                sel = idx[mask]
                per_layer.append((f"{pname}{li}", relative_error(analytic[sel], numeric[sel])))
            start += p.size
    return GradCheckResult(relative_error(analytic[idx], numeric[idx]), per_layer, analytic, numeric, idx)
