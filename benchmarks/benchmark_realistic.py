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
    indices = torch.randint(n_nu1, (n_basis, polynomial_order), dtype=torch.int16, device=device)
    multipliers = torch.rand((n_basis,), dtype=dtype, device=device)

    # Warm-up:
    for _ in range(10):
        optimized_implementation(nu1_basis, indices, multipliers)

    # Benchmark:
    start = time.time()
    for _ in range(1000):
        optimized_implementation(nu1_basis, indices, multipliers)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    print(f"Execution time FW (optimized): {end-start} ms")

    nu1_basis.requires_grad_(True)

    # Warm-up:
    for _ in range(10):
        atomic_energies = optimized_implementation(nu1_basis, indices, multipliers)
        total_energy = torch.sum(atomic_energies)
        total_energy.backward()

    # Benchmark:
    start = time.time()
    for _ in range(1000):
        atomic_energies = optimized_implementation(nu1_basis, indices, multipliers)
        total_energy = torch.sum(atomic_energies)
        total_energy.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    print(f"Execution time FW+BW (optimized): {end-start} ms")

    print()


if __name__ == "__main__":
    print()
    benchmark(torch.float64, "cpu")
    benchmark(torch.float32, "cpu")
    if torch.cuda.is_available():
        benchmark(torch.float64, "cuda")
        benchmark(torch.float32, "cuda")

