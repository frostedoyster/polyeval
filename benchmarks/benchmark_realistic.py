import torch
from polyeval.lib import reference_implementation, optimized_implementation
import time


def benchmark(dtype, device):

    n_nu1 = 2000
    n_atoms = 1000
    n_basis = 100000
    polynomial_order = 4

    print(f"Benchmarking dtype={dtype} and device={device}")

    nu1_basis = torch.rand((n_atoms, n_nu1), dtype=dtype, device=device)
    indices = torch.randint(n_nu1, (n_basis, polynomial_order), dtype=torch.long, device=device)
    multipliers = torch.rand((n_basis,), dtype=dtype, device=device)

    # Warm-up optimized:
    for _ in range(10):
        optimized_implementation(nu1_basis, indices, multipliers)

    # Benchmark optimized:
    start = time.time()
    for _ in range(1000):
        optimized_implementation(nu1_basis, indices, multipliers)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    print(f"Execution time (optimized): {end-start} ms")

    print()


if __name__ == "__main__":
    print()
    benchmark(torch.float64, "cpu")
    benchmark(torch.float32, "cpu")
    if torch.cuda.is_available():
        benchmark(torch.float64, "cuda")
        benchmark(torch.float32, "cuda")

