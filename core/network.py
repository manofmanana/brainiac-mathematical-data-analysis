"""
network.py
-----------
High-level Neural Network model combining Dense and Activation layers.
Handles forward propagation, loss computation, and backpropagation.
"""

import numpy as np
from core.layers import Dense, Activation
from typing import List, Tuple, Dict, Optional


class NeuralNetwork:
    """A simple sequential neural network with monitoring for gradient norms.

    The network is small and educational (not optimized for performance). It supports
    selecting activation functions and initialization schemes per Dense layer.
    """
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', init_scheme: str = 'he', seed: Optional[int] = None):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.init_scheme = init_scheme
        self.seed = seed
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            # Dense expects (in_features, out_features) per our constructor
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], init_scheme, seed))
            if i < len(layer_sizes) - 2:
                self.layers.append(Activation(activation))

        self.loss_history: List[float] = []
        # gradient norm tracking per epoch: list of dicts per epoch
        self.grad_norms_per_epoch: List[Dict[str, float]] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def _compute_grad_norms(self) -> Dict[str, float]:
        norms = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'dW'):
                norms[f'dW_l{idx}'] = float(np.linalg.norm(layer.dW))
            if hasattr(layer, 'db'):
                norms[f'db_l{idx}'] = float(np.linalg.norm(layer.db))
        return norms

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, lr: float):
        grad = 2 * (y_pred - y_true) / max(1, y_true.shape[1])
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                if isinstance(layer, Dense):
                    grad = layer.backward(grad, lr)
                else:
                    grad = layer.backward(grad)

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000, verbose: bool = True, callback: Optional[callable] = None):
        self.loss_history = []
        self.grad_norms_per_epoch = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y_pred, y, lr)
            self.loss_history.append(loss)
            self.grad_norms_per_epoch.append(self._compute_grad_norms())
            if callback is not None:
                callback(epoch, loss)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


def run_convergence_experiments(X: np.ndarray, y: np.ndarray, config_list: List[Tuple[str, str]], layer_sizes: List[int], lr: float = 0.01, epochs: int = 300) -> Dict[str, Dict]:
    """Run training for different (activation, init) configs and return loss histories and grad norms.

    Args:
        X, y: training data
        config_list: list of (activation, init_scheme)
        layer_sizes: network architecture
        lr, epochs: training params

    Returns a mapping from config-name to results {loss_history, grad_norms_per_epoch}
    """
    results = {}
    for act, init in config_list:
        name = f"{act}_{init}"
        net = NeuralNetwork(layer_sizes=layer_sizes, activation=act, init_scheme=init)
        net.train(X, y, lr=lr, epochs=epochs, verbose=False)
        results[name] = {
            'loss_history': net.loss_history,
            'grad_norms_per_epoch': net.grad_norms_per_epoch
        }
    return results
