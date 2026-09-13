# Quickstart

```python
import nexora as nx

x = nx.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).mean()
loss.backward()
print(x.grad)
```
