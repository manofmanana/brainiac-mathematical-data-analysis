import numpy as np


def hutchinson_trace(f, params, v_samples=20, eps=1e-4):
    """Estimate trace(H) for scalar function f(params) using Hutchinson estimator.

    f: callable taking flat params and returning scalar
    params: numpy array of parameters (flat)
    v_samples: number of random probe vectors
    eps: finite-difference step for directional second derivative
    """
    n = params.size
    def directional_second_derivative(v):
        p_plus = params + eps * v
        p_minus = params - eps * v
        f1 = f(p_plus)
        f2 = f(params)
        f3 = f(p_minus)
        # second derivative approx: (f(p+eps v) - 2 f(p) + f(p-eps v)) / eps^2
        return (f1 - 2 * f2 + f3) / (eps ** 2)

    total = 0.0
    for _ in range(v_samples):
        v = np.random.choice([-1.0, 1.0], size=n)
        total += directional_second_derivative(v)
    return float(total / v_samples)
