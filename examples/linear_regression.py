import nexora as nx
from nexora import nn, optim

nx.manual_seed(0)
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

x = nx.tensor([[1.0], [2.0], [3.0], [4.0]])
y = nx.tensor([[2.0], [4.0], [6.0], [8.0]])

for _ in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

print("loss:", loss_fn(model(x), y).item())
