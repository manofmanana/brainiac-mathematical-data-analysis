import numpy as np
import pytest

from brainiac import DATASETS, NeuralNetwork, load_dataset
from brainiac.gradcheck import gradient_check


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu", "leaky_relu"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_full_network_gradient_check(activation, task, rng):
    X = rng.normal(size=(3, 25))
    if task == "classification":
        Y = np.eye(3)[:, rng.integers(0, 3, 25)]
        sizes = [3, 6, 5, 3]
    else:
        Y = rng.normal(size=(1, 25))
        sizes = [3, 6, 5, 1]
    net = NeuralNetwork(sizes, activation, "xavier", task, seed=4)
    result = gradient_check(net, X, Y)
    assert result.overall < 1e-6, result.per_layer
    assert all(err < 1e-5 for _, err in result.per_layer)


def test_gradient_check_does_not_change_parameters(moons):
    net = NeuralNetwork([2, 4, 2], task="classification", seed=0)
    before = net.get_flat_params().copy()
    gradient_check(net, moons.X_train[:, :20], moons.Y_train[:, :20])
    np.testing.assert_array_equal(before, net.get_flat_params())


def test_flat_params_roundtrip():
    net = NeuralNetwork([3, 5, 2], seed=0)
    theta = np.arange(net.n_params, dtype=float)
    net.set_flat_params(theta)
    np.testing.assert_array_equal(net.get_flat_params(), theta)
    assert net.n_params == 3 * 5 + 5 + 5 * 2 + 2


def test_learning_rate_is_not_divided_twice():
    # One SGD step must move w by exactly lr * dL/dw (the old code divided by N twice).
    net = NeuralNetwork([1, 1], "linear", "normal", "regression", seed=0)
    net.set_flat_params(np.array([0.5, 0.0]))
    X, Y = np.array([[2.0]]), np.array([[3.0]])
    net.fit(X, Y, epochs=1, lr=0.1, optimizer="sgd")
    # L = (2w - 3)^2, dL/dw = 4(2w - 3) = -8 at w = 0.5, so w becomes 0.5 + 0.8
    assert net.get_flat_params()[0] == pytest.approx(1.3)


@pytest.mark.parametrize("name", list(DATASETS))
def test_training_learns_every_dataset(name):
    data = load_dataset(name)
    net = NeuralNetwork([data.n_features, 16, 16, data.n_outputs], "tanh", "xavier", data.task, seed=0)
    hist = net.fit(data.X_train, data.Y_train, epochs=150, lr=0.01, batch_size=32)
    assert hist.loss[-1] < 0.5 * hist.loss[0]
    if data.task == "classification":
        assert net.accuracy(data.X_test, data.Y_test) > 0.9
    else:
        assert net.loss(data.X_test, data.Y_test) < 0.1


def test_training_is_reproducible(moons):
    def run():
        net = NeuralNetwork([2, 8, 2], task="classification", seed=7)
        return net.fit(moons.X_train, moons.Y_train, epochs=20, batch_size=16).loss

    assert run() == run()


def test_history_tracks_grad_norms_per_dense_layer(moons):
    net = NeuralNetwork([2, 8, 8, 2], task="classification", seed=0)
    hist = net.fit(moons.X_train, moons.Y_train, epochs=5)
    assert len(hist.grad_norms) == 5
    assert all(len(row) == 3 for row in hist.grad_norms)
    assert len(hist.accuracy) == 5


def test_divergence_stops_early(moons):
    net = NeuralNetwork([2, 8, 2], "relu", "he", "regression", seed=0)
    Y = moons.Y_train[:1] * 1e3
    with np.errstate(all="ignore"):
        hist = net.fit(moons.X_train, Y, epochs=200, lr=10.0, optimizer="sgd")
    assert len(hist.loss) < 200
    assert not np.isfinite(hist.loss[-1])


def test_invalid_configuration():
    with pytest.raises(ValueError):
        NeuralNetwork([3])
    with pytest.raises(ValueError):
        NeuralNetwork([3, 1], task="clustering")
