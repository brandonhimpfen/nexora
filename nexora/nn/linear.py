import math
import numpy as np
from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-bound, bound, (in_features, out_features)).astype(np.float32))
        self.bias = Parameter(np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out
