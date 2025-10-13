import numpy as np


def numerical_grad(f, x, eps=1e-6):
    """Compute numerical gradient of scalar function f at vector x.

    f: function that accepts x (numpy array) and returns scalar
    x: numpy array (will be flattened)
    returns: gradient array of same shape as x
    """
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fx1 = f(x.copy())
        x[idx] = orig - eps
        fx2 = f(x.copy())
        grad[idx] = (fx1 - fx2) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad
