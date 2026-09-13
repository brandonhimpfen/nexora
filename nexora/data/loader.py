import numpy as np
from nexora.tensor import Tensor


def _collate(items):
    first = items[0]
    if isinstance(first, Tensor):
        return Tensor(np.stack([x.data for x in items]))
    if isinstance(first, np.ndarray):
        return Tensor(np.stack(items))
    if isinstance(first, (tuple, list)):
        cols = list(zip(*items))
        return tuple(_collate(list(col)) for col in cols)
    return Tensor(np.asarray(items))


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, seed=None, drop_last=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            yield _collate([self.dataset[int(i)] for i in batch_indices])

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
