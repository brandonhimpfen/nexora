import nexora as nx
from nexora import nn


def test_cross_entropy_backward():
    logits = nx.tensor([[2.0, 1.0, 0.0], [0.5, 2.0, 0.2]], requires_grad=True)
    target = nx.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(logits, target)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad.shape == logits.shape
