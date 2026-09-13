import numpy as np
import nexora as nx


def test_tensor_basics_and_broadcasting():
    x = nx.tensor([[1.0], [2.0]], requires_grad=True)
    y = nx.tensor([10.0, 20.0, 30.0], requires_grad=True)
    z = x + y
    assert z.shape == (2, 3)
    np.testing.assert_allclose(z.numpy(), [[11, 21, 31], [12, 22, 32]])
    z.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[3.0], [3.0]])
    np.testing.assert_allclose(y.grad.numpy(), [2.0, 2.0, 2.0])


def test_reshape_transpose_indexing():
    base = nx.arange(6, dtype=np.float32, requires_grad=True)
    x = base.reshape(2, 3)
    y = x.T[1]
    np.testing.assert_allclose(y.numpy(), [1, 4])
    y.sum().backward()
    np.testing.assert_allclose(base.grad.numpy(), [0, 1, 0, 0, 1, 0])
