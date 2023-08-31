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

    nu1_basis_ref = torch.rand((n_atoms, n_nu1), dtype=dtype, device=device, requires_grad=True)
    indices_ref = torch.randint(n_nu1, (n_basis, polynomial_order), dtype=torch.int16, device=device)
    multipliers_ref = torch.rand((n_basis,), dtype=dtype, device=device)

    nu1_basis_opt = nu1_basis_ref.clone().detach().requires_grad_(True)
    indices_opt = indices_ref.clone()
    multipliers_opt = multipliers_ref.clone()

    atomic_energies_ref = reference_implementation(nu1_basis_ref, indices_ref, multipliers_ref)
    atomic_energies_opt = optimized_implementation(nu1_basis_opt, indices_opt, multipliers_opt)

    assert torch.allclose(
        atomic_energies_ref,
        atomic_energies_opt
    )

    total_energy_ref = torch.sum(atomic_energies_ref)
    total_energy_opt = torch.sum(atomic_energies_opt)

    total_energy_ref.backward()
    total_energy_opt.backward()

    assert torch.allclose(
        nu1_basis_ref.grad,
        nu1_basis_opt.grad
    )

    print("Tests passed successfully!")


if __name__ == "__main__":
    test(torch.float64, "cpu")
    test(torch.float32, "cpu")
    if torch.cuda.is_available():
        test(torch.float64, "cuda")
        test(torch.float32, "cuda")
