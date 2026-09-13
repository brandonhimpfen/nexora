import numpy as np
from .backend import CPU
from .device import device as make_device
from .autograd.engine import backward as run_backward


def _ensure_tensor(value):
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, dtype=None, requires_grad=False, device="cpu"):
        self.device = make_device(device)
        self.data = CPU.array(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        if self.requires_grad and not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError("Gradients are supported only for floating-point tensors in Nexora 0.1.")
        self.grad = None
        self._ctx = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_leaf(self):
        return self._ctx is None

    @property
    def T(self):
        return self.transpose()

    def numpy(self):
        return self.data.copy()

    def item(self):
        return self.data.item()

    def detach(self):
        return Tensor(self.data, requires_grad=False, device=self.device)

    def backward(self, gradient=None, retain_graph=False):
        return run_backward(self, gradient=gradient, retain_graph=retain_graph)

    def zero_grad(self):
        self.grad = None

    def reshape(self, *shape):
        from .ops import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, tuple(shape))

    def transpose(self, *axes):
        from .ops import Transpose
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        return Transpose.apply(self, axes)

    def squeeze(self, axis=None):
        from .ops import Squeeze
        return Squeeze.apply(self, axis)

    def unsqueeze(self, axis):
        from .ops import Unsqueeze
        return Unsqueeze.apply(self, axis)

    def sum(self, axis=None, keepdims=False):
        from .ops import Sum
        return Sum.apply(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        from .ops import Mean
        return Mean.apply(self, axis, keepdims)

    def exp(self):
        from .ops import Exp
        return Exp.apply(self)

    def log(self):
        from .ops import Log
        return Log.apply(self)

    def relu(self):
        from .ops import ReLU
        return ReLU.apply(self)

    def sigmoid(self):
        from .ops import Sigmoid
        return Sigmoid.apply(self)

    def tanh(self):
        from .ops import Tanh
        return Tanh.apply(self)

    def __getitem__(self, index):
        from .ops import GetItem
        return GetItem.apply(self, index)

    def __add__(self, other):
        from .ops import Add
        return Add.apply(self, _ensure_tensor(other))

    __radd__ = __add__

    def __sub__(self, other):
        from .ops import Sub
        return Sub.apply(self, _ensure_tensor(other))

    def __rsub__(self, other):
        return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other):
        from .ops import Mul
        return Mul.apply(self, _ensure_tensor(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        from .ops import Div
        return Div.apply(self, _ensure_tensor(other))

    def __rtruediv__(self, other):
        return _ensure_tensor(other).__truediv__(self)

    def __pow__(self, power):
        from .ops import Pow
        return Pow.apply(self, power)

    def __neg__(self):
        from .ops import Neg
        return Neg.apply(self)

    def __matmul__(self, other):
        from .ops import MatMul
        return MatMul.apply(self, _ensure_tensor(other))

    def __repr__(self):
        attrs = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({self.data!r}, dtype={self.dtype}, device='{self.device}'{attrs})"
