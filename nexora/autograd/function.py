from .context import Context


class Function:
    @classmethod
    def apply(cls, *inputs, **kwargs):
        from nexora.tensor import Tensor

        ctx = Context()
        raw_inputs = [x.data if isinstance(x, Tensor) else x for x in inputs]
        out = cls.forward(ctx, *raw_inputs, **kwargs)
        requires_grad = any(isinstance(x, Tensor) and x.requires_grad for x in inputs)
        result = Tensor(out, requires_grad=requires_grad)
        if requires_grad:
            result._ctx = (cls, ctx, inputs)
        return result

    @staticmethod
    def forward(ctx, *inputs, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
