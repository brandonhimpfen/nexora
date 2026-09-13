from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-2, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [None] * len(self.parameters)

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            grad = param.grad.data
            if self.momentum:
                if self._velocity[i] is None:
                    self._velocity[i] = grad.copy()
                else:
                    self._velocity[i] = self.momentum * self._velocity[i] + grad
                grad = self._velocity[i]
            param.data -= self.lr * grad
