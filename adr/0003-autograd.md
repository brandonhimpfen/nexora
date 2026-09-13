# ADR 0003: autograd

**Status:** Accepted

## Context

Nexora 0.1 requires a clear, minimal architectural decision for autograd.

## Decision

Use the architecture described in the project contract: eager execution is canonical, CPU/NumPy is the reference implementation, dynamic reverse-mode autograd is explicit and inspectable, state serialization avoids arbitrary Python object loading, and future hardware support remains behind a backend boundary.

## Consequences

The 0.1 codebase remains small and understandable. Future systems must preserve these semantics or explicitly approve a breaking change before 1.0.
