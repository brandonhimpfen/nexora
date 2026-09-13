# Autograd

Nexora constructs a dynamic reverse-mode differentiation graph during eager execution. Call `backward()` on a scalar result, or provide an explicit gradient for a non-scalar output. Leaf gradients accumulate in `.grad` until cleared.
