r"""Network layers with explicit forward and backward passes.

For a batch :math:`X \in \mathbb{R}^{d \times N}` (one sample per column), a
dense layer computes :math:`Z = W X + b\mathbf{1}^\top`. Given the upstream
gradient :math:`G = \partial L / \partial Z`, the chain rule gives

.. math::
    \frac{\partial L}{\partial W} = G X^\top, \qquad
    \frac{\partial L}{\partial b} = G \mathbf{1}, \qquad
    \frac{\partial L}{\partial X} = W^\top G.

Layers only *compute* gradients; updating parameters is the optimizer's job.
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation
from .initializers import init_weights


class Dense:
    def __init__(self, fan_in: int, fan_out: int, init: str = "he", rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng()
        self.W = init_weights(fan_in, fan_out, init, rng)
        self.b = np.zeros((fan_out, 1))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.W.shape[1]:
            raise ValueError(f"Dense layer expects {self.W.shape[1]} input features, got {x.shape[0]}")
        self._x = x
        return self.W @ x + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._x is not None, "forward() must run before backward()"
        # grad_out already carries the 1/N from the loss, so these are exact
        # gradients of the mean loss: no further rescaling.
        self.dW[...] = grad_out @ self._x.T
        self.db[...] = grad_out.sum(axis=1, keepdims=True)
        return self.W.T @ grad_out


class Activation:
    params: list[np.ndarray] = []
    grads: list[np.ndarray] = []

    def __init__(self, name: str = "relu"):
        self.fn = get_activation(name)
        self.name = name
        self._z: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._z = z
        return self.fn.f(z)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._z is not None, "forward() must run before backward()"
        return grad_out * self.fn.df(self._z)
