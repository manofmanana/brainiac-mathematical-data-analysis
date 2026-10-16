"""A small, fully explicit feed-forward neural network.

Data uses the column convention from the maths: ``X`` has shape
``(n_features, n_samples)``. Targets are ``(n_outputs, n_samples)``; for
classification they are one-hot encoded and the network outputs raw logits
(softmax is folded into the cross-entropy loss for stability).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .layers import Activation, Dense
from .losses import LOSSES, softmax
from .optimizers import make_optimizer


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    # grad_norms[epoch][layer] = ||dL/dW_layer||_F on the full training set
    grad_norms: list[list[float]] = field(default_factory=list)


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        init: str = "he",
        task: str = "regression",
        seed: int | None = 0,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and an output size")
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        self.layer_sizes = list(layer_sizes)
        self.activation, self.init, self.task, self.seed = activation, init, task, seed
        self.loss_fn = LOSSES["cross_entropy" if task == "classification" else "mse"]

        rng = np.random.default_rng(seed)
        self.layers: list[Dense | Activation] = []
        for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], strict=True)):
            self.layers.append(Dense(fan_in, fan_out, init, rng))
            if i < len(layer_sizes) - 2:  # no activation after the output layer
                self.layers.append(Activation(activation))

    # ----------------------------------------------------------------- basics
    @property
    def dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]

    @property
    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params]

    @property
    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads]

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def loss_and_grad(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Forward + backward pass. Returns the mean loss; gradients land in ``self.grads``."""
        loss, grad = self.loss_fn(self.forward(X), Y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.loss_fn(self.forward(X), Y)[0]

    # ------------------------------------------------- flat parameter vectors
    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.params])

    def set_flat_params(self, theta: np.ndarray) -> None:
        i = 0
        for p in self.params:
            p[...] = theta[i : i + p.size].reshape(p.shape)
            i += p.size

    def get_flat_grads(self) -> np.ndarray:
        return np.concatenate([g.ravel() for g in self.grads])

    def flat_loss(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
        """Loss as a function of a flat parameter vector (restores the original params)."""
        original = self.get_flat_params()
        self.set_flat_params(theta)
        try:
            return self.loss(X, Y)
        finally:
            self.set_flat_params(original)

    def flat_grad(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        original = self.get_flat_params()
        self.set_flat_params(theta)
        try:
            self.loss_and_grad(X, Y)
            return self.get_flat_grads().copy()
        finally:
            self.set_flat_params(original)

    # --------------------------------------------------------------- training
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.05,
        optimizer: str = "adam",
        batch_size: int | None = None,
        callback: Callable[[int, float], None] | None = None,
    ) -> History:
        """Mini-batch training. ``batch_size=None`` means full-batch gradient descent."""
        opt = make_optimizer(optimizer, lr)
        n = X.shape[1]
        bs = n if not batch_size or batch_size >= n else int(batch_size)
        rng = np.random.default_rng(None if self.seed is None else self.seed + 1)
        history = History()

        for epoch in range(epochs):
            order = rng.permutation(n) if bs < n else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                self.loss_and_grad(X[:, idx], Y[:, idx])
                opt.step(self.params, self.grads)

            # Full-batch diagnostics after each epoch.
            loss = self.loss_and_grad(X, Y)
            history.loss.append(loss)
            history.grad_norms.append([float(np.linalg.norm(d.dW)) for d in self.dense_layers])
            if self.task == "classification":
                history.accuracy.append(self.accuracy(X, Y))
            if not np.isfinite(loss):
                break  # diverged; keep the history up to here
            if callback is not None:
                callback(epoch, loss)
        return history

    # ------------------------------------------------------------- inference
    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        return out.argmax(axis=0) if self.task == "classification" else out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only defined for classification")
        return softmax(self.forward(X))

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == Y.argmax(axis=0)))
