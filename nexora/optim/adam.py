import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            g = param.grad.data
            if self.weight_decay:
                g = g + self.weight_decay * param.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        self.decoupled_weight_decay = weight_decay

    def step(self):
        if self.decoupled_weight_decay:
            for param in self.parameters:
                param.data *= 1 - self.lr * self.decoupled_weight_decay
        super().step()
