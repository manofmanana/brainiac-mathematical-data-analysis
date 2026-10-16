"""Brainiac: a neural network built from first principles in NumPy.

Every derivative is written by hand and verified numerically; see the
``gradcheck`` and ``curvature`` modules for the verification tools.
"""

from .activations import ACTIVATIONS
from .datasets import DATASETS, Dataset, load_dataset
from .initializers import INIT_SCHEMES
from .network import History, NeuralNetwork
from .optimizers import OPTIMIZERS

__all__ = [
    "ACTIVATIONS",
    "DATASETS",
    "INIT_SCHEMES",
    "OPTIMIZERS",
    "Dataset",
    "History",
    "NeuralNetwork",
    "load_dataset",
]

__version__ = "2.0.0"
