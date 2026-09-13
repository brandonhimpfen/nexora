import numpy as np
from nexora.autograd import Function


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.meta["input_shape"] = a.shape
        return np.reshape(a, shape)

    @staticmethod
    def backward(ctx, grad_output):
        return np.reshape(grad_output, ctx.meta["input_shape"]), None


class Transpose(Function):
    @staticmethod
    def forward(ctx, a, axes=None):
        ctx.meta["axes"] = axes
        return np.transpose(a, axes=axes)

    @staticmethod
    def backward(ctx, grad_output):
        axes = ctx.meta["axes"]
        if axes is None:
            return np.transpose(grad_output), None
        inverse = np.argsort(axes)
        return np.transpose(grad_output, axes=inverse), None


class Squeeze(Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        ctx.meta["input_shape"] = a.shape
        return np.squeeze(a, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        return np.reshape(grad_output, ctx.meta["input_shape"]), None


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, a, axis):
        ctx.meta["axis"] = axis
        return np.expand_dims(a, axis)

    @staticmethod
    def backward(ctx, grad_output):
        return np.squeeze(grad_output, axis=ctx.meta["axis"]), None
