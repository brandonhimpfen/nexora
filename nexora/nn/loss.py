import numpy as np
from .module import Module
from nexora.tensor import Tensor


class MSELoss(Module):
    def forward(self, prediction, target):
        if not isinstance(target, Tensor):
            target = Tensor(target)
        diff = prediction - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        # Numerically stable log-softmax expressed through differentiable Nexora ops.
        target_arr = target.data if isinstance(target, Tensor) else np.asarray(target)
        # A detached max shift is valid because softmax is invariant to additive constants.
        shift = Tensor(np.max(logits.data, axis=-1, keepdims=True))
        shifted = logits - shift
        log_probs = shifted - shifted.exp().sum(axis=-1, keepdims=True).log()
        rows = np.arange(target_arr.shape[0])
        chosen = log_probs[rows, target_arr.astype(np.int64)]
        return -chosen.mean()
