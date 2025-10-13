"""Small script to run convergence experiments across activations and init schemes.
This is intended to be run as a script or imported by notebooks.
"""
import numpy as np
from core.network import NeuralNetwork, run_convergence_experiments


def simple_compare(X, y, layer_sizes, activations, inits, lr=0.01, epochs=200):
    configs = [(a, i) for a in activations for i in inits]
    results = run_convergence_experiments(X, y, configs, layer_sizes=layer_sizes, lr=lr, epochs=epochs)
    return results


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 200))
    true_w = rng.normal(size=(1, 3))
    y = (true_w @ X).reshape(1, -1)

    acts = ['relu', 'gelu']
    inits = ['he', 'xavier']
    res = simple_compare(X, y, [3, 16, 1], acts, inits, lr=0.02, epochs=120)
    for k, v in res.items():
        print(k, 'final loss', v['loss_history'][-1])
