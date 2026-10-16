r"""Curvature of the loss through Hessian-vector products.

The Hessian :math:`H = \nabla^2 L(\theta)` is never formed. Instead we use

.. math::
    Hv \approx \frac{\nabla L(\theta + \varepsilon v) - \nabla L(\theta - \varepsilon v)}{2\varepsilon},

which costs two gradient evaluations. From that:

* **Hutchinson's estimator**: for Rademacher :math:`v` (entries :math:`\pm 1`),
  :math:`\mathbb{E}[v^\top H v] = \operatorname{tr}(H)`.
* **Power iteration**: :math:`v \leftarrow Hv / \lVert Hv \rVert` converges to the
  eigenvector with the largest-magnitude eigenvalue, :math:`\lambda = v^\top H v`.
  Large :math:`\lambda_{\max}` means a sharp minimum; gradient descent is only
  stable when :math:`\eta < 2 / \lambda_{\max}`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

GradFn = Callable[[np.ndarray], np.ndarray]


def hessian_vector_product(grad_fn: GradFn, theta: np.ndarray, v: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    scale = eps / max(np.linalg.norm(v), 1e-12)
    return (grad_fn(theta + scale * v) - grad_fn(theta - scale * v)) / (2.0 * scale)


@dataclass
class TraceEstimate:
    mean: float
    stderr: float
    samples: np.ndarray


def hutchinson_trace(grad_fn: GradFn, theta: np.ndarray, n_samples: int = 30, seed: int = 0) -> TraceEstimate:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples)
    for i in range(n_samples):
        v = rng.choice([-1.0, 1.0], size=theta.size)
        samples[i] = v @ hessian_vector_product(grad_fn, theta, v)
    return TraceEstimate(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n_samples)), samples)


def top_eigenvalue(grad_fn: GradFn, theta: np.ndarray, iters: int = 50, tol: float = 1e-6, seed: int = 0) -> float:
    """Largest-magnitude Hessian eigenvalue by power iteration."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(theta.size)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        hv = hessian_vector_product(grad_fn, theta, v)
        new_lam = float(v @ hv)
        norm = np.linalg.norm(hv)
        if norm < 1e-12:
            return 0.0
        v = hv / norm
        if abs(new_lam - lam) <= tol * max(1.0, abs(new_lam)):
            return new_lam
        lam = new_lam
    return lam
