import tempfile
import nexora as nx
from nexora import nn

model = nn.Linear(3, 2)
with tempfile.NamedTemporaryFile(suffix=".nxstate") as f:
    nx.save(model.state_dict(), f.name)
    state = nx.load(f.name)
    model.load_state_dict(state)
print("state restored")
