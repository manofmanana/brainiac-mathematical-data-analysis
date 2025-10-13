"""
plotting.py
------------
Plotting helpers for the app. Exposes:
- plot_loss(loss_history)
- plot_gradient_heatmap(grad_norms_per_epoch)
- plot_convergence_comparison(results_dict)
- plot_3d_loss_surface(dense_layer, X, y, weight_idx_pair=(0,1), grid_scale=0.5)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_loss(loss_history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(loss_history)), loss_history, color="#00ff88")
    ax.set_title("Loss Over Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    return fig


def plot_gradient_heatmap(grad_norms_per_epoch):
    """Create a heatmap where rows are epochs and columns are tracked gradient names.

    grad_norms_per_epoch: list of dicts {name: value}
    """
    if not grad_norms_per_epoch:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'No gradient data', ha='center')
        return fig

    # build matrix
    keys = sorted(grad_norms_per_epoch[0].keys())
    data = np.array([[epoch.get(k, 0.0) for k in keys] for epoch in grad_norms_per_epoch])
    fig, ax = plt.subplots(figsize=(8, max(3, len(keys) * 0.3 + 2)))
    sns.heatmap(data.T, cmap='viridis', ax=ax)
    ax.set_yticks(np.arange(len(keys)) + 0.5)
    ax.set_yticklabels(keys)
    ax.set_xlabel('Epoch')
    ax.set_title('Gradient Norms (per epoch)')
    return fig


def plot_convergence_comparison(results_dict):
    """Plot loss curves for multiple experiment runs.

    results_dict: {name: {'loss_history': [...], ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results_dict.items():
        lh = res.get('loss_history', [])
        ax.plot(range(len(lh)), lh, label=name)
    ax.set_title('Convergence Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_3d_loss_surface(dense_layer, X, y, weight_idx_pair=(0, 1), grid_scale=0.5):
    """Sweep two weights in a dense layer across a grid and compute loss surface.

    dense_layer: instance of Dense (we'll vary W.flat[idx_a], W.flat[idx_b])
    X, y: data to compute loss on
    weight_idx_pair: tuple of indices into flattened W to sweep
    grid_scale: relative range scale around current weight values

    Returns a matplotlib Figure with a 3D surface.
    """
    W = dense_layer.W.copy()
    flat = W.flatten()
    idx_a, idx_b = weight_idx_pair
    a0, b0 = float(flat[idx_a]), float(flat[idx_b])
    # grid ranges
    a_range = np.linspace(a0 - grid_scale, a0 + grid_scale, 60)
    b_range = np.linspace(b0 - grid_scale, b0 + grid_scale, 60)
    A, B = np.meshgrid(a_range, b_range)
    Z = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            flat_copy = flat.copy()
            flat_copy[idx_a] = A[i, j]
            flat_copy[idx_b] = B[i, j]
            W_try = flat_copy.reshape(W.shape)
            # compute forward through this single dense layer then squared-error loss
            preds = W_try @ X + dense_layer.b
            Z[i, j] = np.mean((preds - y) ** 2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel(f'W[{idx_a}]')
    ax.set_ylabel(f'W[{idx_b}]')
    ax.set_zlabel('Loss')
    ax.set_title('3D Loss Surface (two-weight sweep)')
    return fig
