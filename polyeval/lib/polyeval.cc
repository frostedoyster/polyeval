#include <iostream>
#include <vector>
#include <chrono>
#include <torch/extension.h>


template <typename scalar_t>
torch::Tensor forward_t(torch::Tensor nu1_basis, torch::Tensor indices, torch::Tensor multipliers) {
    long n_monomials = indices.size(1);
    long n_atoms = nu1_basis.size(0);
    long n_nu1 = nu1_basis.size(1);
    long n_basis = indices.size(0);

    scalar_t* nu1_basis_ptr = nu1_basis.data_ptr<scalar_t>();
    scalar_t* multipliers_ptr = multipliers.data_ptr<scalar_t>();
    long* indices_ptr = indices.data_ptr<long>();

    torch::Tensor atomic_energies = torch::empty({n_atoms}, torch::TensorOptions().dtype(nu1_basis.dtype()));
    scalar_t* atomic_energies_ptr = atomic_energies.data_ptr<scalar_t>();

    // auto start = std::chrono::high_resolution_clock::now();

    # pragma omp parallel for
    for (long i_atom = 0; i_atom < n_atoms; i_atom++) {
        scalar_t result = 0.0;
        long i_atom_shift = i_atom*n_nu1;
        for (long i_basis = 0; i_basis < n_basis; i_basis++) {
            long i_basis_shift = n_monomials*i_basis;
            scalar_t temp = multipliers_ptr[i_basis];
            for (long i_monomial = 0; i_monomial < n_monomials; i_monomial++) {
                temp *= nu1_basis_ptr[i_atom_shift+indices_ptr[i_basis_shift+i_monomial]];
            }
            result += temp;
        }
        atomic_energies_ptr[i_atom] = result;
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    return atomic_energies;
}


class Polyeval : public torch::autograd::Function<Polyeval> {

public:

    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor nu1_basis,
        torch::Tensor indices,
        torch::Tensor multipliers
    ) {

        // Dispatch type by hand
        if (nu1_basis.dtype() == c10::kDouble) {
            return forward_t<double>(nu1_basis, indices, multipliers);
        } else if (nu1_basis.dtype() == c10::kFloat) {
            return forward_t<float>(nu1_basis, indices, multipliers);
        } else {
            throw std::runtime_error("Unsupported dtype");
        }
    }

    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outputs) {

        throw std::runtime_error("not implemented");
    }
};


torch::Tensor polyeval(torch::Tensor nu1_basis, torch::Tensor indices, torch::Tensor multipliers) {
    return Polyeval::apply(nu1_basis, indices, multipliers);
}


TORCH_LIBRARY(polyeval_cc, m) {
    m.def("polyeval", &polyeval);
}

