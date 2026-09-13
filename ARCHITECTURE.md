# Architecture

Nexora 0.1 uses eager execution as canonical semantics.

```text
User code
   |
   v
Tensor / nn / optim / data
   |
   v
Operations <--> Autograd
   |
   v
Backend
   |
   v
CPU / NumPy
```

Dependency direction is downward. Tensor and autograd do not depend on neural-network, optimizer, data, or serialization modules. The only backend in 0.1 is CPU/NumPy. Future accelerator backends must preserve eager tensor semantics.

The project deliberately does not contain placeholder compiler, IR, distributed, serving, or quantization packages.
