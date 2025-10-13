"""
Generate demo figures for README by running a tiny training demo and a small convergence experiment.
Saves: notebooks/figures/loss_curve.png,
       notebooks/figures/grad_heatmap.png,
       notebooks/figures/convergence_comparison.png

Run from project root with the project venv Python, e.g.:
/Users/alejandroines/mathematical-mind/venv/bin/python notebooks/generate_figures.py
"""

from pathlib import Path
import sys

# Make project root importable (this script lives in notebooks/)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import traceback

try:
    from core.layers import Dense
    from core.network import NeuralNetwork
    from utils.plotting import plot_loss, plot_gradient_heatmap, plot_convergence_comparison
except Exception as e:
    print('Failed to import project modules. Make sure you run this from the project root and the venv is active.')
    traceback.print_exc()
    raise


def main():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 200))
    true_w = rng.normal(size=(1, 3))
    y = (true_w @ X).reshape(1, -1)

    net = NeuralNetwork(layer_sizes=[3, 16, 1], activation='relu', init_scheme='he', seed=0)
    net.train(X, y, lr=0.05, epochs=120, verbose=False)

    fig1 = plot_loss(net.loss_history)
    fig2 = plot_gradient_heatmap(net.grad_norms_per_epoch)

    fig_dir = Path(__file__).resolve().parent / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig1_path = fig_dir / 'loss_curve.png'
    fig2_path = fig_dir / 'grad_heatmap.png'
    fig1.savefig(fig1_path, bbox_inches='tight')
    fig2.savefig(fig2_path, bbox_inches='tight')
    print('Saved', fig1_path, fig2_path)

    # convergence experiment
    try:
        from experiments.run_convergence import simple_compare
        acts = ['relu', 'gelu']
        inits = ['he', 'xavier']
        res = simple_compare(X, y, [3, 16, 1], acts, inits, lr=0.02, epochs=120)
        fig3 = plot_convergence_comparison(res)
        fig3_path = fig_dir / 'convergence_comparison.png'
        fig3.savefig(fig3_path, bbox_inches='tight')
        print('Saved', fig3_path)
    except Exception:
        print('Convergence experiment failed (optional). See traceback:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
