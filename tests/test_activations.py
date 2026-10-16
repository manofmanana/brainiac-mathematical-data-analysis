import numpy as np
import pytest

from brainiac.activations import ACTIVATIONS, get_activation
from brainiac.gradcheck import numerical_gradient


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_derivative_matches_finite_differences(name, rng):
    act = get_activation(name)
    x = rng.normal(scale=2.0, size=50)
    x = x[np.abs(x) > 1e-3]  # avoid the ReLU kink at exactly 0
    numeric = numerical_gradient(lambda v: float(np.sum(act.f(v))), x, eps=1e-6)
    np.testing.assert_allclose(act.df(x), numeric, rtol=1e-6, atol=1e-8)


def test_gelu_derivative_known_values():
    # d/dx x*Phi(x) = Phi(x) + x*phi(x); at x = 1 that is 0.841345 + 0.241971
    gelu = get_activation("gelu")
    assert gelu.df(np.array([0.0]))[0] == pytest.approx(0.5)
    assert gelu.df(np.array([1.0]))[0] == pytest.approx(1.0833155, rel=1e-6)


def test_sigmoid_is_stable_for_extreme_inputs():
    s = get_activation("sigmoid").f(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(s))


def test_unknown_activation():
    with pytest.raises(ValueError):
        get_activation("swish9000")
