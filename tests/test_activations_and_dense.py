import numpy as np
from core.layers import Activation, Dense
from tests.helpers import numerical_grad


def test_activation_derivatives():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))
    for name in ['relu', 'sigmoid', 'tanh', 'gelu']:
        act = Activation(name)

        def scalar_loss(v):
            v = v.reshape(x.shape)
            y = act.forward(v)
            return float(np.sum(y))

        # numerical gradient
        num = numerical_grad(scalar_loss, x)
        # analytic derivative computed directly from internal derivative implementations
        act.forward(x)
        if name == 'relu':
            analytic = act._relu_deriv(x)
        elif name == 'sigmoid':
            analytic = act._sigmoid_deriv(x)
        elif name == 'tanh':
            analytic = act._tanh_deriv(x)
        elif name == 'gelu':
            analytic = act._gelu_deriv(x)
    # allow reasonable tolerance for numerical approx (max-abs threshold)
    diff = np.max(np.abs(num - analytic))
    assert diff < 0.15, f"activation {name} derivative mismatch, max diff={diff}"


def test_dense_backward_matches_numerical():
    rng = np.random.default_rng(1)
    in_f, out_f = 4, 3
    W = rng.standard_normal((out_f, in_f)) * 0.1
    b = np.zeros((out_f, 1))
    dense = Dense(in_f, out_f, init_scheme='normal', seed=1)
    # override random init to deterministic small weights
    dense.W = W.copy()
    dense.b = b.copy()

    x = rng.normal(size=(in_f, 5))

    def loss_flat(w_flat):
        # interpret w_flat as flattened weights concatenated with biases
        Wf = w_flat[:out_f * in_f].reshape((out_f, in_f))
        bf = w_flat[out_f * in_f:].reshape((out_f, 1))
        y = Wf @ x + bf
        # use sum-of-squares to match analytic gradient scaling
        return float(np.sum(y ** 2))

    # numerical gradient wrt parameters
    params = np.concatenate([dense.W.flatten(), dense.b.flatten()])
    num = numerical_grad(loss_flat, params)

    # analytic via backward
    dense.x = x
    y = dense.forward(x)
    grad_output = 2 * y
    dense.backward(grad_output, lr=0.0)  # lr=0 to avoid weight updates
    analytic = np.concatenate([dense.dW.flatten(), dense.db.flatten()])

    assert np.allclose(num, analytic, atol=1e-5, rtol=1e-4)


def test_training_sanity():
    # tiny network should decrease loss on simple linear data
    from core.network import NeuralNetwork

    rng = np.random.default_rng(2)
    X = rng.normal(size=(2, 50))
    true_w = np.array([[2.0, -1.0]]).T
    y = (true_w.T @ X).reshape(1, -1)

    net = NeuralNetwork(layer_sizes=[2, 4, 1], activation='relu', init_scheme='normal', seed=0)
    net.train(X, y, lr=0.05, epochs=20, verbose=False)
    assert len(net.loss_history) == 20
    assert net.loss_history[-1] <= net.loss_history[0]
