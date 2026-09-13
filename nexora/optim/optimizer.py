class Optimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
