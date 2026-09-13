import nexora as nx
from nexora import nn, optim


def test_sgd_reduces_simple_loss():
    nx.manual_seed(0)
    model = nn.Linear(1, 1)
    opt = optim.SGD(model.parameters(), lr=0.05)
    x = nx.tensor([[1.0], [2.0], [3.0]])
    y = nx.tensor([[2.0], [4.0], [6.0]])
    loss_fn = nn.MSELoss()
    before = loss_fn(model(x), y).item()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    after = loss_fn(model(x), y).item()
    assert after < before


def test_adamw_step_runs():
    p = nn.Parameter([1.0])
    p.grad = nx.tensor([0.5])
    opt = optim.AdamW([p], lr=0.01)
    old = p.item()
    opt.step()
    assert p.item() != old
