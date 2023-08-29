import torch


def python_implementation(nu1_basis, indices, multipliers):

    polynomial_order = indices.shape[1]

    product = nu1_basis.index_select(1, indices[:, 0])
    for monomial_index in range(2, polynomial_order):
        product *= nu1_basis.index_select(1, indices[:, monomial_index-1])
    atomic_energies = product @ multipliers

    return atomic_energies
