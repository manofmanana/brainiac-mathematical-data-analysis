import numpy as np
import pytest

from brainiac import load_dataset


@pytest.fixture(scope="session")
def moons():
    return load_dataset("Two moons")


@pytest.fixture
def rng():
    return np.random.default_rng(0)
