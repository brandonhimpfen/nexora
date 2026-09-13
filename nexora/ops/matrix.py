import numpy as np
from nexora.autograd import Function
from nexora.errors import ShapeMismatchError


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        try:
            out = np.matmul(a, b)
        except ValueError as exc:
            raise ShapeMismatchError(
                "Matrix multiplication failed.\n\n"
                f"left:  {a.shape}\nright: {b.shape}\n\n"
                "Expected compatible inner dimensions."
            ) from exc
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.matmul(grad_output, np.swapaxes(b, -1, -2))
        grad_b = np.matmul(np.swapaxes(a, -1, -2), grad_output)
        return grad_a, grad_b
