import numpy as np
from nexora.autograd import Function


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a - b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output / b, -grad_output * a / (b ** 2)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, power):
        ctx.save_for_backward(a)
        ctx.meta["power"] = power
        return a ** power

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        power = ctx.meta["power"]
        return grad_output * power * (a ** (power - 1)), None


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output / a
