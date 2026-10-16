"""Build and execute notebooks/brainiac_walkthrough.ipynb.

    python notebooks/build_notebook.py

Keeping the notebook generated from this script means it always matches the
library and always ships with fresh, real outputs.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
    md(
        "# Brainiac walkthrough: backpropagation, verified\n\n"
        "This notebook trains a network built from scratch in NumPy (`brainiac/`), proves the hand-written "
        "gradients are correct, and runs a small controlled experiment. The interactive version is the Streamlit app."
    ),
    code(
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from brainiac import NeuralNetwork, load_dataset\n"
        "from brainiac.gradcheck import gradient_check\n"
        "from brainiac.curvature import top_eigenvalue, hutchinson_trace\n"
        "from brainiac.experiments import compare_configs\n"
        "plt.rcParams['figure.figsize'] = (7, 4)"
    ),
    md(
        "## 1. Data\n\nTwo interleaving half-moons with Gaussian noise; features are standardised with "
        "training-set statistics only."
    ),
    code(
        "data = load_dataset('Two moons')\n"
        "print('train', data.X_train.shape, 'test', data.X_test.shape)\n"
        "print('train mean', data.X_train.mean(axis=1).round(6), 'train std', data.X_train.std(axis=1).round(6))"
    ),
    md(
        "## 2. Backprop is correct\n\n"
        "Compare the analytic gradient with central finite differences, "
        "$\\frac{L(\\theta+\\varepsilon e_i)-L(\\theta-\\varepsilon e_i)}{2\\varepsilon}$, for **every** parameter."
    ),
    code(
        "net = NeuralNetwork([2, 16, 16, 2], activation='tanh', init='xavier', task='classification', seed=0)\n"
        "result = gradient_check(net, data.X_train[:, :64], data.Y_train[:, :64])\n"
        "print(f'overall relative error: {result.overall:.2e}  ({result.checked.size} parameters)')\n"
        "for name, err in result.per_layer:\n"
        "    print(f'  {name:3s} {err:.2e}')"
    ),
    md("## 3. Train"),
    code(
        "hist = net.fit(data.X_train, data.Y_train, epochs=200, lr=0.01, optimizer='adam', batch_size=32)\n"
        "print(f'final training loss {hist.loss[-1]:.4f}')\n"
        "print(f'test accuracy {net.accuracy(data.X_test, data.Y_test):.1%}')\n"
        "fig, ax = plt.subplots()\n"
        "ax.semilogy(hist.loss)\n"
        "ax.set(xlabel='epoch', ylabel='cross-entropy (log)', title='Training loss')\n"
        "plt.show()"
    ),
    md(
        "## 4. Curvature at the solution\n\n"
        "Power iteration on Hessian-vector products gives $\\lambda_{\\max}$; gradient descent is only stable "
        "for $\\eta < 2/\\lambda_{\\max}$."
    ),
    code(
        "g = lambda th: net.flat_grad(th, data.X_train, data.Y_train)\n"
        "theta = net.get_flat_params()\n"
        "lam = top_eigenvalue(g, theta)\n"
        "tr = hutchinson_trace(g, theta, n_samples=50)\n"
        "print(f'lambda_max = {lam:.3f}  ->  stable SGD step size < {2 / lam:.3f}')\n"
        "print(f'trace(H) = {tr.mean:.2f} +/- {tr.stderr:.2f}')"
    ),
    md(
        "## 5. Experiment: activation x initialisation, 3 seeds each\n\n"
        "A deeper, narrower network (four hidden layers of 8) makes the vanishing-gradient effect visible."
    ),
    code(
        "runs = compare_configs(data, [8, 8, 8, 8], ['relu', 'tanh', 'sigmoid'], ['he', 'xavier'], seeds=[0, 1, 2],\n"
        "                       epochs=150, lr=0.01, target=0.1)\n"
        "fig, ax = plt.subplots()\n"
        "for r in runs:\n"
        "    ax.semilogy(np.exp(np.nanmean(np.log(r.curves), axis=0)), label=r.label)\n"
        "ax.set(xlabel='epoch', ylabel='loss (geometric mean over seeds)', title='Convergence')\n"
        "ax.legend(fontsize=8)\n"
        "plt.show()\n"
        "for r in sorted(runs, key=lambda r: r.final_loss.mean()):\n"
        "    print(f'{r.label:18s} final loss {r.final_loss.mean():.4f} +/- {r.final_loss.std():.4f}   '\n"
        "          f'test acc {r.test_metric.mean():.1%}')"
    ),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
out = HERE / "brainiac_walkthrough.ipynb"
nbformat.write(nb, out)
print("wrote", out)
