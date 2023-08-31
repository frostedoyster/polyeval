import torch


def reference_implementation(nu1_basis, indices, multipliers):

    polynomial_order = indices.shape[1]
    indices = indices.to(torch.long)

    product = nu1_basis.index_select(1, indices[:, 0])
    for monomial_index in range(1, polynomial_order):
        product *= nu1_basis.index_select(1, indices[:, monomial_index])
    atomic_energies = product @ multipliers

    return atomic_energies
