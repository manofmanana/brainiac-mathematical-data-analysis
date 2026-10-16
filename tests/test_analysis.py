import numpy as np
import pytest

from brainiac import NeuralNetwork, load_dataset
from brainiac.curvature import hessian_vector_product, hutchinson_trace, top_eigenvalue
from brainiac.datasets import one_hot
from brainiac.experiments import compare_configs
from brainiac.landscape import interpolation_path, loss_surface
from brainiac.linalg_lab import fft_filter, pca, svd_compress

A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 1.0]])


def quad_grad(theta):
    return A @ theta  # gradient of 0.5 * theta^T A theta, whose Hessian is A


def test_hvp_is_exact_on_a_quadratic(rng):
    v = rng.normal(size=3)
    np.testing.assert_allclose(hessian_vector_product(quad_grad, np.ones(3), v), A @ v, rtol=1e-6)


def test_hutchinson_trace_and_power_iteration():
    est = hutchinson_trace(quad_grad, np.zeros(3), n_samples=400, seed=1)
    assert est.mean == pytest.approx(np.trace(A), abs=4 * est.stderr + 1e-6)
    assert top_eigenvalue(quad_grad, np.zeros(3), iters=200) == pytest.approx(np.linalg.eigvalsh(A).max(), rel=1e-4)


def test_loss_surface_is_centred_on_trained_params(moons):
    net = NeuralNetwork([2, 8, 2], task="classification", seed=0)
    net.fit(moons.X_train, moons.Y_train, epochs=100, lr=0.02)
    before = net.get_flat_params().copy()
    s = loss_surface(net, moons.X_train, moons.Y_train, resolution=11)
    assert s.loss.shape == (11, 11)
    assert s.loss[5, 5] == pytest.approx(net.loss(moons.X_train, moons.Y_train))
    np.testing.assert_array_equal(before, net.get_flat_params())


def test_interpolation_path_endpoints(moons):
    net = NeuralNetwork([2, 8, 2], task="classification", seed=0)
    theta0 = net.get_flat_params().copy()
    loss0 = net.loss(moons.X_train, moons.Y_train)
    net.fit(moons.X_train, moons.Y_train, epochs=50)
    _, losses = interpolation_path(net, theta0, moons.X_train, moons.Y_train, n=11, extend=0.0)
    assert losses[0] == pytest.approx(loss0)
    assert losses[-1] == pytest.approx(net.loss(moons.X_train, moons.Y_train))


def test_compare_configs_summarises_seeds(moons):
    runs = compare_configs(moons, [8], ["relu", "tanh"], ["he"], seeds=[0, 1], epochs=10, target=0.5)
    assert [r.label for r in runs] == ["relu + he", "tanh + he"]
    assert runs[0].curves.shape == (2, 10)
    assert np.all((runs[0].test_metric >= 0) & (runs[0].test_metric <= 1))


def test_datasets_are_standardised_without_test_leakage():
    d = load_dataset("Iris (4 features, 3 species)")
    np.testing.assert_allclose(d.X_train.mean(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(d.X_train.std(axis=1), 1, atol=1e-6)
    assert d.X_train.shape[1] + d.X_test.shape[1] == 150
    assert not np.allclose(d.X_test.mean(axis=1), 0, atol=1e-10)


def test_one_hot():
    np.testing.assert_array_equal(one_hot(np.array([2, 0]), 3), [[0, 1], [0, 0], [1, 0]])


def test_svd_compression_eckart_young(rng):
    img = rng.random((40, 30))
    full = svd_compress(img, 30)
    assert full.energy == pytest.approx(1.0)
    assert full.relative_error == pytest.approx(0.0, abs=1e-6)
    low = svd_compress(img, 5)
    s = low.singular_values
    assert low.relative_error == pytest.approx(np.sqrt(np.sum(s[5:] ** 2) / np.sum(s**2)))


def test_pca_eigenvalues_match_numpy_covariance():
    d = load_dataset("Iris (4 features, 3 species)")
    res = pca(d.X_train)
    np.testing.assert_allclose(np.sort(res.eigenvalues), np.sort(np.linalg.eigvalsh(np.cov(d.X_train))), rtol=1e-10)
    assert res.explained_ratio.sum() == pytest.approx(1.0)
    assert res.projection.shape == (2, d.X_train.shape[1])


def test_fft_lowpass_with_full_radius_is_identity(rng):
    img = rng.random((32, 32))
    out, _ = fft_filter(img, radius=2.0, mode="low")
    np.testing.assert_allclose(out, img, atol=1e-10)
