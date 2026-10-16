import numpy as np
import pytest

from brainiac.gradcheck import numerical_gradient
from brainiac.initializers import INIT_SCHEMES, init_weights, target_std
from brainiac.layers import Dense
from brainiac.losses import mse, softmax, softmax_cross_entropy
from brainiac.optimizers import SGD, Adam, Momentum


@pytest.mark.parametrize("scheme", INIT_SCHEMES)
def test_initializer_variance(scheme, rng):
    W = init_weights(400, 300, scheme, rng)
    assert W.shape == (300, 400)
    assert abs(W.mean()) < 0.01
    assert W.std() == pytest.approx(target_std(scheme, 400, 300), rel=0.03)


def test_xavier_and_lecun_differ():
    # The old code used sqrt(1/fan_in) for both.
    assert target_std("xavier", 100, 10) != pytest.approx(target_std("lecun", 100, 10))
    assert target_std("xavier", 100, 10) == pytest.approx(np.sqrt(2 / 110))


def test_dense_backward_matches_numerical(rng):
    layer = Dense(4, 3, "xavier", rng)
    x = rng.normal(size=(4, 7))
    out = layer.forward(x)
    layer.backward(2 * out)  # dL/dZ for L = sum(Z^2)

    def f(w):
        return float(np.sum((w.reshape(3, 4) @ x + layer.b) ** 2))

    np.testing.assert_allclose(layer.dW.ravel(), numerical_gradient(f, layer.W.ravel()), rtol=1e-6)
    np.testing.assert_allclose(layer.db.ravel(), (2 * out).sum(axis=1))


def test_dense_rejects_wrong_input_size(rng):
    with pytest.raises(ValueError):
        Dense(3, 2, rng=rng).forward(np.zeros((4, 5)))


def test_mse_value_and_gradient():
    loss, grad = mse(np.array([[1.0, 3.0]]), np.array([[0.0, 1.0]]))
    assert loss == pytest.approx((1 + 4) / 2)
    np.testing.assert_allclose(grad, [[1.0, 2.0]])


def test_softmax_cross_entropy_is_stable_and_correct():
    logits = np.array([[1000.0, 0.0], [0.0, 0.0], [-1000.0, 0.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    loss, grad = softmax_cross_entropy(logits, y)
    assert np.isfinite(loss)
    assert loss == pytest.approx((0.0 + np.log(3)) / 2)
    np.testing.assert_allclose(grad, (softmax(logits) - y) / 2)
    np.testing.assert_allclose(softmax(logits).sum(axis=0), 1.0)


@pytest.mark.parametrize("opt_cls", [SGD, Momentum, Adam])
def test_optimizers_minimise_a_quadratic(opt_cls):
    A = np.diag([1.0, 10.0])
    theta = np.array([3.0, -2.0])
    opt = opt_cls(lr=0.05)
    for _ in range(500):
        opt.step([theta], [A @ theta])
    assert np.linalg.norm(theta) < 1e-3


def test_adam_first_step_has_size_lr():
    # With bias correction the very first Adam step is ~lr * sign(g), whatever |g| is.
    theta = np.array([1.0, -1.0])
    Adam(lr=0.1).step([theta], [np.array([123.0, -0.001])])
    np.testing.assert_allclose(theta, [0.9, -0.9], atol=1e-6)
