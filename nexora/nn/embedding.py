import numpy as np
from .module import Module
from .parameter import Parameter


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        scale = 1.0 / max(1, embedding_dim) ** 0.5
        self.weight = Parameter(
            np.random.normal(0.0, scale, (num_embeddings, embedding_dim)).astype(np.float32)
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices):
        idx = indices.data if hasattr(indices, "data") else indices
        return self.weight[idx.astype(np.int64)]
