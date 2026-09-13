import numpy as np
import nexora as nx


def test_scalar_grad():
    x = nx.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    assert abs(x.grad.item() - 6.0) < 1e-6


def test_chain_rule():
    x = nx.tensor([1.5, -2.0, 0.5], requires_grad=True)
    y = ((x * 2.0) ** 2).sum()
    y.backward()
    np.testing.assert_allclose(x.grad.numpy(), 8 * x.numpy())


def test_finite_difference():
    x0 = np.array([0.3, -0.8, 1.2], dtype=np.float64)
    x = nx.tensor(x0.copy(), requires_grad=True)
    loss = (x.tanh() * x).sum()
    loss.backward()
    eps = 1e-6
    numeric = np.zeros_like(x0)
    for i in range(len(x0)):
        p = x0.copy(); p[i] += eps
        m = x0.copy(); m[i] -= eps
        fp = np.sum(np.tanh(p) * p)
        fm = np.sum(np.tanh(m) * m)
        numeric[i] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(x.grad.numpy(), numeric, rtol=1e-5, atol=1e-6)
