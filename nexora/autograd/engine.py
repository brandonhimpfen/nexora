import numpy as np
from nexora.errors import GradientError


def unbroadcast(grad, shape):
    grad = np.asarray(grad)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def topological_sort(root):
    visited = set()
    order = []

    def visit(t):
        ident = id(t)
        if ident in visited:
            return
        visited.add(ident)
        if t._ctx is not None:
            _, _, inputs = t._ctx
            for item in inputs:
                if hasattr(item, "_ctx"):
                    visit(item)
        order.append(t)

    visit(root)
    return order


def backward(root, gradient=None, retain_graph=False):
    if not root.requires_grad:
        raise GradientError("Cannot call backward() on a tensor that does not require gradients.")

    if gradient is None:
        if root.data.size != 1:
            raise GradientError(
                "backward() requires an explicit gradient for non-scalar tensors."
            )
        gradient = np.ones_like(root.data)
    elif hasattr(gradient, "data"):
        gradient = gradient.data
    else:
        gradient = np.asarray(gradient, dtype=root.data.dtype)

    if gradient.shape != root.data.shape:
        raise GradientError(
            f"Gradient shape {gradient.shape} does not match tensor shape {root.data.shape}."
        )

    grads = {id(root): gradient}
    order = topological_sort(root)

    for tensor in reversed(order):
        grad_out = grads.get(id(tensor))
        if grad_out is None:
            continue

        if tensor.is_leaf and tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = tensor.__class__(grad_out.copy(), requires_grad=False)
            else:
                tensor.grad.data = tensor.grad.data + grad_out

        if tensor._ctx is None:
            continue

        fn, ctx, inputs = tensor._ctx
        input_grads = fn.backward(ctx, grad_out)
        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)

        if len(input_grads) != len(inputs):
            raise GradientError(
                f"{fn.__name__}.backward returned {len(input_grads)} gradients for "
                f"{len(inputs)} inputs."
            )

        for input_obj, grad in zip(inputs, input_grads):
            if grad is None or not hasattr(input_obj, "requires_grad") or not input_obj.requires_grad:
                continue
            grad = unbroadcast(grad, input_obj.shape)
            key = id(input_obj)
            grads[key] = grad if key not in grads else grads[key] + grad

        if not retain_graph:
            tensor._ctx = None
