import numpy as np
from nexora.autograd import Function


def _normalize_axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a if a >= 0 else ndim + a for a in axis)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.meta.update(axis=axis, keepdims=keepdims, input_shape=a.shape)
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.meta["input_shape"]
        axis = ctx.meta["axis"]
        keepdims = ctx.meta["keepdims"]
        grad = grad_output
        if axis is not None and not keepdims:
            for ax in sorted(_normalize_axes(axis, len(shape))):
                grad = np.expand_dims(grad, ax)
        return np.broadcast_to(grad, shape), None, None


class Mean(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        axes = _normalize_axes(axis, a.ndim)
        denom = np.prod([a.shape[x] for x in axes]) if axis is not None else a.size
        ctx.meta.update(axis=axis, keepdims=keepdims, input_shape=a.shape, denom=denom)
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.meta["input_shape"]
        axis = ctx.meta["axis"]
        keepdims = ctx.meta["keepdims"]
        grad = grad_output
        if axis is not None and not keepdims:
            for ax in sorted(_normalize_axes(axis, len(shape))):
                grad = np.expand_dims(grad, ax)
        return np.broadcast_to(grad, shape) / ctx.meta["denom"], None, None
