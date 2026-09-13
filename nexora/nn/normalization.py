import numpy as np
from .module import Module
from .parameter import Parameter


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
        self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))

    def forward(self, x):
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mean = x.mean(axis=axes, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=axes, keepdims=True)
        normalized = centered / ((var + self.eps) ** 0.5)
        return normalized * self.weight + self.bias
