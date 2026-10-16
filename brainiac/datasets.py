r"""Datasets, all generated locally (no network access) and fully seeded.

Features are standardised with training-set statistics,
:math:`x' = (x - \mu_{\text{train}}) / \sigma_{\text{train}}`, and the same
transform is applied to the test split so no test information leaks into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class Dataset:
    name: str
    task: str  # "classification" | "regression"
    X_train: np.ndarray  # (d, N_train), standardised
    Y_train: np.ndarray  # one-hot (k, N) or targets (1, N)
    X_test: np.ndarray
    Y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    mean: np.ndarray
    std: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[0]

    @property
    def n_outputs(self) -> int:
        return self.Y_train.shape[0]

    @property
    def labels_train(self) -> np.ndarray:
        return self.Y_train.argmax(axis=0)

    @property
    def labels_test(self) -> np.ndarray:
        return self.Y_test.argmax(axis=0)


def one_hot(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((k, labels.size))
    out[labels, np.arange(labels.size)] = 1.0
    return out


def _split_and_scale(name, task, X, targets, feature_names, class_names, test_frac, rng) -> Dataset:
    """X: (N, d) rows; targets: (N,) labels or (N,) real values."""
    n = X.shape[0]
    order = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test, train = order[:n_test], order[n_test:]
    mean, std = X[train].mean(axis=0), X[train].std(axis=0) + 1e-12
    Xs = (X - mean) / std
    if task == "classification":
        Y = one_hot(targets.astype(int), len(class_names))
    else:
        Y = targets.reshape(1, -1).astype(float)
    return Dataset(name, task, Xs[train].T, Y[:, train], Xs[test].T, Y[:, test], feature_names, class_names, mean, std)


def make_moons(n: int = 400, noise: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    X += rng.normal(scale=noise, size=X.shape)
    return X, np.r_[np.zeros(n1, int), np.ones(n - n1, int)]


def make_circles(n: int = 400, noise: float = 0.08, factor: float = 0.45, seed: int = 0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.r_[np.ones(n1), np.full(n - n1, factor)]
    X = np.c_[r * np.cos(t), r * np.sin(t)] + rng.normal(scale=noise, size=(n, 2))
    return X, np.r_[np.zeros(n1, int), np.ones(n - n1, int)]


def make_spirals(n: int = 450, noise: float = 0.15, classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    per = n // classes
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, per)
        t = np.linspace(c * 2 * np.pi / classes, c * 2 * np.pi / classes + 3.5, per) + rng.normal(scale=noise, size=per)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(per, c))
    return np.vstack(X), np.concatenate(y)


def load_iris_arrays():
    raw = np.genfromtxt(DATA_DIR / "iris.csv", delimiter=",", skip_header=1, dtype=str)
    X = raw[:, :4].astype(float)
    names, y = np.unique(raw[:, 4], return_inverse=True)
    return X, y, ["sepal length", "sepal width", "petal length", "petal width"], [str(s) for s in names]


def make_sine(n: int = 300, noise: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, n)
    return x.reshape(-1, 1), np.sin(x) + 0.3 * np.sin(3 * x) + rng.normal(scale=noise, size=n)


DATASETS = {
    "Two moons": "classification",
    "Concentric circles": "classification",
    "Spirals (3 classes)": "classification",
    "Iris (4 features, 3 species)": "classification",
    "Noisy sine wave (regression)": "regression",
}


def load_dataset(name: str, noise: float | None = None, seed: int = 0, test_frac: float = 0.25) -> Dataset:
    rng = np.random.default_rng(seed + 1000)
    xy_names = ["x₁", "x₂"]
    if name == "Two moons":
        X, y = make_moons(noise=0.15 if noise is None else noise, seed=seed)
        return _split_and_scale(name, "classification", X, y, xy_names, ["moon A", "moon B"], test_frac, rng)
    if name == "Concentric circles":
        X, y = make_circles(noise=0.08 if noise is None else noise, seed=seed)
        return _split_and_scale(name, "classification", X, y, xy_names, ["outer", "inner"], test_frac, rng)
    if name == "Spirals (3 classes)":
        X, y = make_spirals(noise=0.15 if noise is None else noise, seed=seed)
        return _split_and_scale(name, "classification", X, y, xy_names, ["arm 1", "arm 2", "arm 3"], test_frac, rng)
    if name.startswith("Iris"):
        X, y, feats, classes = load_iris_arrays()
        if noise:
            X = X + np.random.default_rng(seed).normal(scale=noise, size=X.shape)
        return _split_and_scale(name, "classification", X, y, feats, classes, test_frac, rng)
    if name.startswith("Noisy sine"):
        X, t = make_sine(noise=0.1 if noise is None else noise, seed=seed)
        return _split_and_scale(name, "regression", X, t, ["x"], [], test_frac, rng)
    raise ValueError(f"Unknown dataset {name!r}")
