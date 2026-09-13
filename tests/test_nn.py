import numpy as np
import nexora as nx
from nexora import nn


def test_linear_backward_and_parameters():
    nx.manual_seed(1)
    layer = nn.Linear(3, 2)
    x = nx.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = layer(x).sum()
    y.backward()
    params = list(layer.named_parameters())
    assert [name for name, _ in params] == ["weight", "bias"]
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_sequential_training_state():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    model.eval()
    assert model.training is False
    assert all(m.training is False for m in model._modules.values())


def test_embedding_gradient_accumulates_rows():
    emb = nn.Embedding(5, 3)
    idx = nx.tensor([1, 1, 3])
    out = emb(idx).sum()
    out.backward()
    assert emb.weight.grad.shape == emb.weight.shape
    np.testing.assert_allclose(emb.weight.grad.numpy()[1], np.full(3, 2.0))
    np.testing.assert_allclose(emb.weight.grad.numpy()[3], np.full(3, 1.0))


def test_layernorm_output_shape():
    ln = nn.LayerNorm(4)
    x = nx.randn((3, 4), requires_grad=True)
    y = ln(x)
    assert y.shape == (3, 4)
    y.sum().backward()
    assert x.grad.shape == x.shape
