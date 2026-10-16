"""Controlled experiments: compare activation x initialization choices over several seeds.

Reporting the mean and spread over seeds (rather than one lucky run) is what
turns a demo into an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import Dataset
from .network import NeuralNetwork


@dataclass
class RunSummary:
    activation: str
    init: str
    curves: np.ndarray  # (n_seeds, epochs) training loss
    final_loss: np.ndarray  # (n_seeds,)
    test_metric: np.ndarray  # accuracy (classification) or MSE (regression), per seed
    epochs_to_target: np.ndarray  # first epoch below `target` (nan if never)

    @property
    def label(self) -> str:
        return f"{self.activation} + {self.init}"


def compare_configs(
    data: Dataset,
    hidden: list[int],
    activations: list[str],
    inits: list[str],
    seeds: list[int],
    epochs: int = 150,
    lr: float = 0.01,
    optimizer: str = "adam",
    batch_size: int | None = 32,
    target: float | None = None,
) -> list[RunSummary]:
    sizes = [data.n_features, *hidden, data.n_outputs]
    out = []
    for act in activations:
        for init in inits:
            curves, finals, metrics, hits = [], [], [], []
            for seed in seeds:
                net = NeuralNetwork(sizes, act, init, data.task, seed=seed)
                hist = net.fit(
                    data.X_train, data.Y_train, epochs=epochs, lr=lr, optimizer=optimizer, batch_size=batch_size
                )
                loss = np.full(epochs, np.nan)
                loss[: len(hist.loss)] = hist.loss
                curves.append(loss)
                finals.append(hist.loss[-1])
                metrics.append(
                    net.accuracy(data.X_test, data.Y_test)
                    if data.task == "classification"
                    else net.loss(data.X_test, data.Y_test)
                )
                below = np.flatnonzero(loss < target) if target is not None else np.array([])
                hits.append(float(below[0] + 1) if below.size else np.nan)
            out.append(RunSummary(act, init, np.array(curves), np.array(finals), np.array(metrics), np.array(hits)))
    return out
