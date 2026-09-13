import numpy as np
from nexora.autograd import Function


class GetItem(Function):
    @staticmethod
    def forward(ctx, a, index):
        ctx.meta["index"] = index
        ctx.meta["input_shape"] = a.shape
        return a[index]

    @staticmethod
    def backward(ctx, grad_output):
        grad = np.zeros(ctx.meta["input_shape"], dtype=grad_output.dtype)
        np.add.at(grad, ctx.meta["index"], grad_output)
        return grad, None
