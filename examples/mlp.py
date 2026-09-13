import numpy as np
import nexora as nx
from nexora import nn, optim

nx.manual_seed(1)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
loss_fn = nn.CrossEntropyLoss()

x = nx.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = nx.tensor(np.array([0, 1, 1, 0]))

for _ in range(250):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

print("loss:", loss_fn(model(x), y).item())
