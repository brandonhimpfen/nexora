import numpy as np
import nexora as nx
from nexora import nn


class TinyDataset(nx.data.Dataset):
    def __init__(self):
        self.items = [(np.array([i], dtype=np.float32), i) for i in range(5)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_dataloader_batches():
    loader = nx.data.DataLoader(TinyDataset(), batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    x, y = batches[0]
    assert x.shape == (2, 1)
    assert y.shape == (2,)


def test_serialization_roundtrip(tmp_path):
    model = nn.Linear(2, 1)
    state = model.state_dict()
    path = tmp_path / "model.nxstate"
    nx.save(state, path)
    restored = nx.load(path)
    assert list(restored) == list(state)
    for key in state:
        np.testing.assert_allclose(restored[key], state[key])
