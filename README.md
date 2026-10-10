# Nexora

[![Support Open Work](https://img.shields.io/badge/Support-Open%20Work-0A0A0A?style=flat&logo=github)](https://github.com/brandonhimpfen/support) 
[![GitHub Sponsor](https://srv-cdn.himpfen.io/badges/github/github-flat.svg)](https://github.com/sponsors/brandonhimpfen) 
[![Buy Me a Coffee](https://srv-cdn.himpfen.io/badges/buymeacoffee/buymeacoffee-flat.svg)](https://buymeacoffee.com/brandonhimpfen) 
[![Ko-Fi](https://srv-cdn.himpfen.io/badges/kofi/kofi-flat.svg)](https://ko-fi.com/brandonhimpfen) 
[![PayPal](https://srv-cdn.himpfen.io/badges/paypal/paypal-flat.svg)](https://paypal.me/brandonhimpfen)

Nexora is a small, inspectable tensor and automatic differentiation framework for Python.

> Simple by default. Fast when compiled. Portable by design. Hardware neutral. Inspectable everywhere.

Nexora 0.1 intentionally focuses on CPU tensors, eager execution, reverse-mode autograd, a compact neural-network API, optimizers, data loading, and safe state serialization. GPU backends, compilation, an intermediate representation, distributed execution, quantization, serving, and model runtimes are out of scope for this release.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Quickstart

```python
import nexora as nx

x = nx.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()

print(x.grad.item())  # 6.0
```

A small model:

```python
import nexora as nx
from nexora import nn, optim

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

## v0.1 Scope

Included: tensors, NumPy-backed CPU execution, broadcasting, dynamic reverse-mode autograd, `Module`, `Parameter`, `Linear`, `Embedding`, `LayerNorm`, common activations, MSE/cross-entropy loss, SGD, Adam, AdamW, Dataset/DataLoader, state serialization, tests, examples, benchmarks, and architecture records.

Not included: CUDA, ROCm, Metal, WebGPU, distributed execution, compilation, IR/FTIR, quantization, serving, mobile/browser runtimes, and large-model infrastructure.

## Status

Nexora 0.1 is an educational and experimental framework. Its public API may change throughout the 0.x series.
