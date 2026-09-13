import numpy as np
from .tensor import Tensor
from .device import device, Device
from .backend import CPU
from . import nn, optim, data
from .serialization import save, load

__version__ = "0.1.0"

_rng = np.random.default_rng()


def manual_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)
    np.random.seed(seed)


def tensor(data, dtype=None, requires_grad=False, device="cpu"):
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, device=device)


def zeros(shape, dtype=np.float32, requires_grad=False):
    return Tensor(CPU.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(shape, dtype=np.float32, requires_grad=False):
    return Tensor(CPU.ones(shape, dtype=dtype), requires_grad=requires_grad)


def full(shape, value, dtype=np.float32, requires_grad=False):
    return Tensor(CPU.full(shape, value, dtype=dtype), requires_grad=requires_grad)


def arange(*args, dtype=None, requires_grad=False):
    return Tensor(CPU.arange(*args, dtype=dtype), requires_grad=requires_grad)


def randn(shape, dtype=np.float32, requires_grad=False):
    return Tensor(CPU.randn(shape, _rng, dtype=dtype), requires_grad=requires_grad)


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def exp(x):
    return x.exp()


def log(x):
    return x.log()


def matmul(a, b):
    return a @ b


def sum(x, axis=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return x.mean(axis=axis, keepdims=keepdims)


__all__ = [
    "Tensor", "Device", "device", "tensor", "zeros", "ones", "full", "arange", "randn",
    "manual_seed", "relu", "sigmoid", "tanh", "exp", "log", "matmul", "sum", "mean",
    "save", "load", "nn", "optim", "data"
]
