from nexora.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, dtype=None, requires_grad=True):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, dtype=dtype, requires_grad=requires_grad)
