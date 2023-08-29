import torch
import polyeval
import time


def benchmark(dtype, device):

    n_nu1 = 2000
    n_atoms = 1000
    n_basis = 1000  # 100000 if we wanted to be realistic, but 1000 to make the benchmark run fast
    polynomial_order = 4

    print(f"Benchmarking dtype={dtype} and device={device}")

    nu1_basis = torch.rand((n_atoms, n_nu1), dtype=dtype, device=device)
    indices = torch.randint(n_nu1, (n_basis, polynomial_order), dtype=torch.long, device=device)
    multipliers = torch.rand((n_basis,), dtype=dtype, device=device)

    # Warm-up:
    for _ in range(10):
        polyeval.python_implementation(nu1_basis, indices, multipliers)

    # Benchmark:
    start = time.time()
    for _ in range(1000):
        polyeval.python_implementation(nu1_basis, indices, multipliers)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    print(f"Execution time: {end-start} ms")


if __name__ == "__main__":
    benchmark(torch.float64, "cpu")
    benchmark(torch.float32, "cpu")
    benchmark(torch.float64, "cuda")
    benchmark(torch.float32, "cuda")

