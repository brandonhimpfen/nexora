import nexora as nx

x = nx.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print("gradient:", x.grad.item())
