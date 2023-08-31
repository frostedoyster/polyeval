import torch
from polyeval.lib import reference_implementation, optimized_implementation
from time import time

torch.set_printoptions(precision=5)
# def python_implementation(nu1_basis, indices, multipliers):
#     n_atoms = nu1_basis.shape[0]
#     atom_energies = torch.zeros(n_atoms, dtype=nu1_basis.dtype, device=nu1_basis.device)
#     for i_atom in range(n_atoms):
#         result = 0.0
#         i_atom_shift = i_atom*n_nu1
#         for i_basis in range(n_basis):
#             i_basis_shift = polynomial_order*i_basis
#             temp = multipliers_ref[i_basis]
#             for i_monomial in range(polynomial_order):
#                 temp *= nu1_basis_ref[i_atom_shift+indices_ref[i_basis_shift+i_monomial]]

#             result += temp

#         print (i_atom, result)
#         atom_energies[i_atom] = result

#     return atom_energies

dtype = torch.float32
device = 'cuda'
n_nu1 = 2000
n_atoms = 1000
n_basis = 100000  # 100000 if we wanted to be realistic, but 1000 to make the test run fast
polynomial_order = 4

print(f"Testing dtype={dtype} and device={device}")

nu1_basis_ref = torch.rand(
    (n_atoms, n_nu1), dtype=dtype, device=device, requires_grad=True)
indices_ref = torch.randint(
    n_nu1, (n_basis, polynomial_order), dtype=torch.int16, device=device)
multipliers_ref = torch.rand((n_basis,), dtype=dtype, device=device)


cpu = torch.ops.polyeval_cc.polyeval(nu1_basis_ref.cpu().double(
), indices_ref.cpu().long(), multipliers_ref.cpu().double())

indices = indices_ref.transpose(-1, -2).contiguous()

print(indices.shape)

start = time()
for i in range(1000):
    cuda = torch.ops.polyeval_cu.polyeval(nu1_basis_ref, indices, multipliers_ref, 256, 4)
end = time()
torch.cuda.synchronize()

print(end - start)
print(cuda[-1])
print(cpu[-1])
