# Brainiac — Mathematical Neural Network

Brainiac is an educational, math-first implementation of a feed-forward neural network. It was designed as a portfolio piece to demonstrate both theoretical understanding (multivariable calculus, linear algebra, probability/statistics) and practical engineering (numerical stability, testing, reproducible experiments). The project pairs a compact NumPy implementation of network components with an interactive Streamlit dashboard for visualization and experimentation.

This README explains the project's goals, the math it showcases, how the code is organized, how to run and reproduce experiments, and suggestions for how to present this work in an MS application.

Live deployment can be found here: https://brainiac-mathematical.streamlit.app/

## High-level goals

- Provide a minimal, readable implementation of a neural network from first principles (no deep learning frameworks) so reviewers can inspect the math and code directly.
- Make the mathematics explicit: show activation functions, their derivatives, backpropagation logic, and the effects of initialization and learning rate on training dynamics.
- Supply visual diagnostics (loss curves, gradient norms, 3D loss slices) that make model behavior interpretable and teachable.
- Include small reproducible experiments and tests to demonstrate scientific thinking and engineering rigor.

## Mathematical foundations 

1) Multivariable calculus and backpropagation

- The network uses mean-squared error (MSE) as the loss: $L(\theta)=\frac{1}{N}\sum_{i}(y_i - \hat y_i)^2$.
- Gradients are computed analytically using the chain rule. For a Dense layer with parameters $W,b$ and input $x$, the forward output is $z=W x + b$ and the backward pass computes
	- $\frac{\partial L}{\partial W} = (\partial L/\partial z) x^T$ (implemented and demonstrated)
	- $\frac{\partial L}{\partial b} = \sum_i \partial L/\partial z_i$
- The project includes visualization of gradient norms per-epoch, which helps reveal vanishing/exploding gradients and how activations/initialization mitigate those problems.

2) Linear algebra

- Dense layers are matrix multiplications; the code makes the shapes explicit and enforces input-dimension compatibility.
- Brainiac Lab: SVD-based image reconstructions and PCA-like components illustrate how singular vectors capture structure. FFT visualizations demonstrate frequency structure of random fields.

3) Probability & statistics

- Feature/target normalization is used for stable training and to illustrate the effect of scale on optimization.
- Experimental controls include injecting Gaussian noise and comparing how noise affects final MSE and convergence.

4) Numerical considerations and initialization

- The code implements common initialization schemes (He, Xavier/Glorot, LeCun, Normal, Uniform) and shows how they interact with activations (ReLU, Sigmoid, Tanh, GELU).
- There are small experiments (see `experiments/run_convergence.py`) to compare convergence speed and final loss across init/activation combinations.

## What I implemented (quick code tour)

- `app/main.py`: Streamlit interface. Lightweight, lazy imports for heavy modules, interactive controls, Brainiac Lab visualization, and CSS-based theme.
- `core/layers.py`: Dense and Activation classes with explicit forward/backward methods. Activation derivatives implemented for ReLU, Sigmoid, Tanh, and GELU (GELU uses `scipy.special.erf`).
- `core/network.py`: NeuralNetwork wrapper, training loop, and helper for running convergence experiments.
- `utils/plotting.py`: (existing) plotting helpers that return matplotlib Figures for embedding in the Streamlit app.
- `experiments/run_convergence.py`: small script to run activation/init comparisons and print final losses.
- `utils/hessian_utils.py`: a Hutchinson estimator to approximate trace(H) for simple curvature diagnostics.
- `tests/`: unit tests and numerical gradient checks (activation derivatives, Dense backward, and a tiny training sanity test) demonstrating verification practices.

## How to run locally (recommended reproducible steps)

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) run tests to confirm the environment:

```bash
python -m pytest -q
```

4. Run the Streamlit app:

```bash
streamlit run app/main.py
```

5. Use the sidebar to pick data, set layer sizes, activation, initialization, learning rate, and epochs. The app exposes training diagnostics and Brainiac Lab visualizations.

## Reproducing experiments

- Example experiment script: `python experiments/run_convergence.py` — runs a small synthetic experiment comparing activations (`relu`, `gelu`) and initializations (`he`, `xavier`) and prints final losses.
- For reproducibility, most functions accept an RNG seed parameter or use fixed seeds in example experiments and tests.

## Testing & verification

- A small pytest suite exists under `tests/`:
	- Activation derivatives are numerically checked (finite differences) against analytic derivatives.
	- Dense backward pass is compared to numerical gradients computed over flattened parameter vectors.
	- A simple training sanity test verifies loss decreases on a small linear problem.
- Run the tests with `python -m pytest -q`.

## Presentation tips for grad-school applications

- In your CV: include a 2–3 line summary and a link to the repo. Mention the math topics (backprop derivations, SVD art, Hessian trace diagnostics).
- In your statement of purpose (SOP): describe a single experiment (e.g., how initialization + activation affected convergence) and include a short figure (loss curves) exported from the app.
- Interview talking points: explain the backprop derivation for a Dense layer, show a screenshot of gradient norms over epochs, and briefly summarize what you learned from the experiments.

## File layout

- `app/` — Streamlit application and static CSS/assets.
- `core/` — model building blocks: `layers.py`, `network.py`.
- `utils/` — plotting helpers and numerical diagnostics (hessian estimator).
- `experiments/` — reproducible experiment scripts.
- `tests/` — pytest test suite with numerical checks.

## License & contact

This repository is a personal portfolio project. If you use or adapt the code, please include a note in your application or CV. If you'd like help tailoring a one-page summary or a short video demo for your application, contact: your-email@example.com

---

