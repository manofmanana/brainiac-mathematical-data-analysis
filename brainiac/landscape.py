r"""Visualising the loss landscape around a trained network.

A network has hundreds of parameters, so we look at 2-D slices
:math:`L(\theta^* + \alpha\,\delta + \beta\,\eta)` along two random directions.
Following Li et al. (2018), each direction is *filter-normalised*: every
parameter tensor of the direction is rescaled to the norm of the matching
tensor in :math:`\theta^*`, so slices are comparable across architectures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import NeuralNetwork


def filter_normalized_direction(net: NeuralNetwork, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for p in net.params:
        d = rng.standard_normal(p.shape)
        d *= np.linalg.norm(p) / (np.linalg.norm(d) + 1e-12)
        parts.append(d.ravel())
    return np.concatenate(parts)


@dataclass
class Surface:
    alphas: np.ndarray
    betas: np.ndarray
    loss: np.ndarray  # loss[i, j] at (alphas[j], betas[i])


def loss_surface(
    net: NeuralNetwork, X: np.ndarray, Y: np.ndarray, span: float = 1.0, resolution: int = 31, seed: int = 0
) -> Surface:
    rng = np.random.default_rng(seed)
    theta = net.get_flat_params()
    d1, d2 = filter_normalized_direction(net, rng), filter_normalized_direction(net, rng)
    grid = np.linspace(-span, span, resolution)
    Z = np.empty((resolution, resolution))
    for i, b in enumerate(grid):
        for j, a in enumerate(grid):
            Z[i, j] = net.flat_loss(theta + a * d1 + b * d2, X, Y)
    return Surface(grid, grid, Z)


def interpolation_path(
    net: NeuralNetwork, theta_start: np.ndarray, X: np.ndarray, Y: np.ndarray, n: int = 41, extend: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    r"""Loss along the straight line :math:`\theta(t) = (1-t)\theta_0 + t\theta^*`."""
    theta_end = net.get_flat_params()
    ts = np.linspace(-extend, 1.0 + extend, n)
    losses = np.array([net.flat_loss((1 - t) * theta_start + t * theta_end, X, Y) for t in ts])
    return ts, losses
