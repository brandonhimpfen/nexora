import time
import numpy as np
import nexora as nx


def bench(label, fn, iterations=100):
    for _ in range(10):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = (time.perf_counter() - start) / iterations
    print(f"{label:24s} {elapsed * 1e6:10.2f} us/op")


if __name__ == "__main__":
    a = nx.randn((256, 256))
    b = nx.randn((256, 256))
    na, nb = a.numpy(), b.numpy()
    bench("nexora add", lambda: a + b)
    bench("numpy add", lambda: na + nb)
    bench("nexora matmul", lambda: a @ b, 20)
    bench("numpy matmul", lambda: na @ nb, 20)
