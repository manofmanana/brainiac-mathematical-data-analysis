"""
backprop.py
------------
Contains utility functions to numerically verify gradients using finite differences.
Useful for debugging your analytical backpropagation implementation.
"""

import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        x[idx] = old_value + eps
        fx1 = f(x)
        x[idx] = old_value - eps
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * eps)
        x[idx] = old_value
        it.iternext()
    return grad

def gradient_check(model, X, y, eps=1e-5):
    """Compares analytical vs numerical gradients for one batch."""
    y_pred = model.forward(X)
    analytical_loss = model.compute_loss(y_pred, y)

    def loss_fn(W):
        model.layers[0].W = W
        return model.compute_loss(model.forward(X), y)

    W = model.layers[0].W
    num_grad = numerical_gradient(loss_fn, W.copy(), eps)
    model.backward(y_pred, y, lr=0)
    an_grad = model.layers[0].W  # after backward call

    diff = np.linalg.norm(an_grad - num_grad) / (np.linalg.norm(an_grad + num_grad) + 1e-8)
    print(f"Gradient check relative error: {diff:.2e}")
