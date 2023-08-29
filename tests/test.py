import torch
from polyeval.lib import reference_implementation, optimized_implementation


import torch
from polyeval.lib import reference_implementation, optimized_implementation
import time


def test(dtype, device):

    n_nu1 = 2000
    n_atoms = 100
    n_basis = 1000  # 100000 if we wanted to be realistic, but 1000 to make the test run fast
    polynomial_order = 4

    print(f"Testing dtype={dtype} and device={device}")

    nu1_basis = torch.rand((n_atoms, n_nu1), dtype=dtype, device=device)
    indices = torch.randint(n_nu1, (n_basis, polynomial_order), dtype=torch.long, device=device)
    multipliers = torch.rand((n_basis,), dtype=dtype, device=device)

    assert torch.allclose(
        reference_implementation(nu1_basis, indices, multipliers),
        optimized_implementation(nu1_basis, indices, multipliers)
    )

    print("Test passed successfully!")


if __name__ == "__main__":
    test(torch.float64, "cpu")
    test(torch.float32, "cpu")
    if torch.cuda.is_available():
        test(torch.float64, "cuda")
        test(torch.float32, "cuda")
