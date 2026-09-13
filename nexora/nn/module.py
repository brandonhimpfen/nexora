from collections import OrderedDict
from .parameter import Parameter
from nexora.errors import ModuleError


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, prefix=""):
        for name, param in self._parameters.items():
            yield f"{prefix}{name}", param
        for name, module in self._modules.items():
            child_prefix = f"{prefix}{name}."
            yield from module.named_parameters(child_prefix)

    def train(self, mode=True):
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return OrderedDict((name, param.numpy()) for name, param in self.named_parameters())

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        unexpected = set(state) - set(params)
        if missing or unexpected:
            raise ModuleError(
                f"State dict mismatch. Missing keys: {sorted(missing)}; "
                f"unexpected keys: {sorted(unexpected)}"
            )
        for name, param in params.items():
            value = state[name]
            if tuple(value.shape) != param.shape:
                raise ModuleError(
                    f"Shape mismatch for {name}: expected {param.shape}, received {value.shape}."
                )
            param.data[...] = value.astype(param.dtype, copy=False)
        return self
