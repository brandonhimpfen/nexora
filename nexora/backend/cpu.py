import numpy as np
from .base import Backend


class CPUBackend(Backend):
    name = "cpu"

    def array(self, data, dtype=None):
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def full(self, shape, value, dtype=None):
        return np.full(shape, value, dtype=dtype)

    def arange(self, *args, dtype=None):
        return np.arange(*args, dtype=dtype)

    def randn(self, shape, rng, dtype=None):
        arr = rng.standard_normal(shape)
        return arr.astype(dtype or np.float32)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def sum(self, a, axis=None, keepdims=False):
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        return np.mean(a, axis=axis, keepdims=keepdims)


CPU = CPUBackend()
