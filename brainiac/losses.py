"""Loss functions returning the mean loss and its gradient w.r.t. the network output.

Shapes follow the column convention used throughout the package: outputs are
``(k, N)`` with one column per sample.
"""

from __future__ import annotations

import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Mean squared error, :math:`L = \frac{1}{N}\sum_i \lVert \hat y_i - y_i \rVert^2`."""
    n = y_true.shape[1]
    diff = y_pred - y_true
    return float(np.sum(diff**2) / n), 2.0 * diff / n


def softmax(z: np.ndarray) -> np.ndarray:
    """Column-wise softmax, shifted by the max for numerical stability."""
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y_onehot: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Softmax + cross-entropy on raw logits.

    Using the log-sum-exp trick, :math:`-\log \mathrm{softmax}(z)_c = \log\sum_j e^{z_j} - z_c`.
    The gradient simplifies to :math:`(\mathrm{softmax}(z) - y) / N`.
    """
    n = y_onehot.shape[1]
    shifted = logits - logits.max(axis=0, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=0, keepdims=True))
    log_probs = shifted - log_sum_exp
    loss = -float(np.sum(y_onehot * log_probs) / n)
    return loss, (np.exp(log_probs) - y_onehot) / n


LOSSES = {"mse": mse, "cross_entropy": softmax_cross_entropy}
