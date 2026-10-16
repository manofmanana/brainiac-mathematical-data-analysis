r"""First-order optimizers. Each ``step`` updates parameter arrays in place.

* **SGD**:       :math:`\theta \leftarrow \theta - \eta g`
* **Momentum**:  :math:`v \leftarrow \beta v + g,\ \theta \leftarrow \theta - \eta v`
* **Adam**:      bias-corrected first/second moment estimates,
  :math:`\theta \leftarrow \theta - \eta\, \hat m / (\sqrt{\hat v} + \epsilon)`
"""

from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        for p, g in zip(params, grads, strict=True):
            p -= self.lr * g


class Momentum:
    def __init__(self, lr: float = 0.1, beta: float = 0.9):
        self.lr, self.beta = lr, beta
        self._v: list[np.ndarray] | None = None

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self._v is None:
            self._v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self._v, strict=True):
            v *= self.beta
            v += g
            p -= self.lr * v


class Adam:
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1t = 1.0 - self.beta1**self.t
        b2t = 1.0 - self.beta2**self.t
        for p, g, m, v in zip(params, grads, self._m, self._v, strict=True):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / b1t) / (np.sqrt(v / b2t) + self.eps)


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}


def make_optimizer(name: str, lr: float):
    try:
        return OPTIMIZERS[name](lr=lr)
    except KeyError:
        raise ValueError(f"Unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}") from None
