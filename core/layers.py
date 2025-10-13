"""
layers.py
----------
Dense and Activation layers with selectable initialization and activation functions.

Features:
- Dense supports initialization schemes: 'he', 'xavier', 'lecun', 'normal', 'uniform'
- Activation supports: 'relu', 'sigmoid', 'tanh', 'gelu'
- Dense stores last weight/bias gradients to enable gradient-norm tracking

All operations use NumPy.
"""

import numpy as np
import math
from scipy.special import erf
from typing import Optional

def _init_weights(in_features: int, out_features: int, scheme: str = 'he', seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    if scheme == 'he':
        return rng.standard_normal((out_features, in_features)) * np.sqrt(2.0 / in_features)
    elif scheme == 'xavier' or scheme == 'glorot':
        return rng.standard_normal((out_features, in_features)) * np.sqrt(1.0 / in_features)
    elif scheme == 'lecun':
        return rng.standard_normal((out_features, in_features)) * np.sqrt(1.0 / in_features)
    elif scheme == 'normal':
        return rng.normal(loc=0.0, scale=0.05, size=(out_features, in_features))
    elif scheme == 'uniform':
        limit = math.sqrt(3.0 / in_features)
        return rng.uniform(-limit, limit, size=(out_features, in_features))
    else:
        # default to small random normal
        return rng.standard_normal((out_features, in_features)) * np.sqrt(2.0 / in_features)


class Dense:
    def __init__(self, in_features: int, out_features: int, init_scheme: str = 'he', seed: Optional[int] = None):
        """Create Dense layer with specified init scheme.

        Args:
            in_features: number of input features
            out_features: number of output features
            init_scheme: one of 'he','xavier','lecun','normal','uniform'
            seed: optional RNG seed
        """
        self.in_features = in_features
        self.out_features = out_features
        self.W = _init_weights(in_features, out_features, scheme=init_scheme, seed=seed).astype(float)
        self.b = np.zeros((out_features, 1), dtype=float)
        self.x = None
        # store gradients for monitoring
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return self.W @ x + self.b

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        # gradient w.r.t. weights: compute sum-of-gradients (so numerical checks align)
        batch = max(1, self.x.shape[1])
        self.dW = (grad_output @ self.x.T)
        self.db = np.sum(grad_output, axis=1, keepdims=True)
        grad_input = self.W.T @ grad_output
        # Update weights (in-place)
        # scale the gradient update by batch for consistent learning rate semantics
        self.W -= (lr / batch) * self.dW
        self.b -= (lr / batch) * self.db
        return grad_input


class Activation:
    def __init__(self, func: str = 'relu'):
        allowed = ['relu', 'sigmoid', 'tanh', 'gelu']
        if func not in allowed:
            raise ValueError(f"Activation must be one of {allowed}")
        self.func_name = func
        self.x = None

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_deriv(self, x):
        return 1 - np.tanh(x) ** 2

    def _gelu(self, x):
        # Gaussian Error Linear Unit (approximate)
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    def _gelu_deriv(self, x):
        # derivative of GELU (using exact formula components)
        sqrt2 = np.sqrt(2.0)
        erf_term = erf(x / sqrt2)
        pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
        return 0.5 * (1.0 + erf_term) + 0.5 * x * (pdf)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        if self.func_name == 'relu':
            return self._relu(x)
        elif self.func_name == 'sigmoid':
            return self._sigmoid(x)
        elif self.func_name == 'tanh':
            return self._tanh(x)
        elif self.func_name == 'gelu':
            return self._gelu(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.func_name == 'relu':
            return grad_output * self._relu_deriv(self.x)
        elif self.func_name == 'sigmoid':
            return grad_output * self._sigmoid_deriv(self.x)
        elif self.func_name == 'tanh':
            return grad_output * self._tanh_deriv(self.x)
        elif self.func_name == 'gelu':
            return grad_output * self._gelu_deriv(self.x)
