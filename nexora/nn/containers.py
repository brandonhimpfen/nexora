from .module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._sequence = []
        for i, module in enumerate(modules):
            name = str(i)
            setattr(self, name, module)
            self._sequence.append(name)

    def forward(self, x):
        for name in self._sequence:
            x = self._modules[name](x)
        return x
