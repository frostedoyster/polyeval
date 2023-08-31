#include <iostream>
#include <vector>
#include <chrono>
#include <torch/extension.h>


template <typename scalar_t, long n_monomials>
torch::Tensor forward_t(torch::Tensor nu1_basis, torch::Tensor indices, torch::Tensor multipliers) {
    // long n_monomials = indices.size(1);
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
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Execution time: " << duration.count() << " us" << std::endl;

    return atomic_energies;
}


template <typename scalar_t, long n_monomials>
std::vector<torch::Tensor> backward_t(torch::Tensor grad_atomic_energies, torch::Tensor nu1_basis, torch::Tensor indices, torch::Tensor multipliers) {
    // long n_monomials = indices.size(1);
    long n_atoms = nu1_basis.size(0);
    long n_nu1 = nu1_basis.size(1);
    long n_basis = indices.size(0);

    scalar_t* nu1_basis_ptr = nu1_basis.data_ptr<scalar_t>();
    scalar_t* multipliers_ptr = multipliers.data_ptr<scalar_t>();
    long* indices_ptr = indices.data_ptr<long>();
    scalar_t* grad_atomic_energies_ptr = grad_atomic_energies.data_ptr<scalar_t>();

    torch::Tensor grad_nu1_basis = torch::Tensor();

    if (nu1_basis.requires_grad()) {
        // auto start = std::chrono::high_resolution_clock::now();
        grad_nu1_basis = torch::zeros_like(nu1_basis);
        scalar_t* grad_nu1_basis_ptr = grad_nu1_basis.data_ptr<scalar_t>();
        # pragma omp parallel for
        for (long i_atom = 0; i_atom < n_atoms; i_atom++) {
            scalar_t grad_atomic_energy = grad_atomic_energies_ptr[i_atom];
            long i_atom_shift = i_atom*n_nu1;
            for (long i_basis = 0; i_basis < n_basis; i_basis++) {
                long i_basis_shift = n_monomials*i_basis;
                scalar_t base_multiplier = grad_atomic_energy*multipliers_ptr[i_basis];
                for (long i_monomial = 0; i_monomial < n_monomials; i_monomial++) {
                    scalar_t temp = base_multiplier;
                    for (long j_monomial = 0; j_monomial < n_monomials; j_monomial++) {
                        if (j_monomial == i_monomial) continue;
                        temp *= nu1_basis_ptr[i_atom_shift+indices_ptr[i_basis_shift+j_monomial]];
                    }
                    grad_nu1_basis_ptr[i_atom_shift+indices_ptr[i_basis_shift+i_monomial]] += temp;
                }
            }
        }
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "Execution time: " << duration.count() << " us" << std::endl;
    }

    return {grad_nu1_basis, torch::Tensor(), torch::Tensor()};
}


class Polyeval : public torch::autograd::Function<Polyeval> {

public:

    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor nu1_basis,
        torch::Tensor indices,
        torch::Tensor multipliers
    ) {
        if (nu1_basis.requires_grad()) ctx->save_for_backward({nu1_basis, indices, multipliers});

        // Dispatch type by hand
        if (nu1_basis.dtype() == c10::kDouble) {
            if (indices.size(1) == 2) return forward_t<double, 2>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 3) return forward_t<double, 3>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 4) return forward_t<double, 4>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 5) return forward_t<double, 5>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 6) return forward_t<double, 6>(nu1_basis, indices, multipliers);
            throw std::runtime_error("Polynomial order is too high.");   
        } else if (nu1_basis.dtype() == c10::kFloat) {
            if (indices.size(1) == 2) return forward_t<float, 2>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 3) return forward_t<float, 3>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 4) return forward_t<float, 4>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 5) return forward_t<float, 5>(nu1_basis, indices, multipliers);
            if (indices.size(1) == 6) return forward_t<float, 6>(nu1_basis, indices, multipliers); 
            throw std::runtime_error("Polynomial order is too high.");     
        } else {
            throw std::runtime_error("Unsupported dtype");
        }
    }

    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outputs) {

        std::vector<torch::Tensor> saved_variables = ctx->get_saved_variables();
        torch::Tensor nu1_basis = saved_variables[0];
        torch::Tensor indices = saved_variables[1];
        torch::Tensor multipliers = saved_variables[2];
        torch::Tensor grad_atomic_energies = grad_outputs[0].contiguous();

        // Dispatch type by hand
        if (nu1_basis.dtype() == c10::kDouble) {
            if (indices.size(1) == 2) return backward_t<double, 2>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 3) return backward_t<double, 3>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 4) return backward_t<double, 4>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 5) return backward_t<double, 5>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 6) return backward_t<double, 6>(grad_atomic_energies, nu1_basis, indices, multipliers);
            throw std::runtime_error("Polynomial order is too high.");
        } else if (nu1_basis.dtype() == c10::kFloat) {
            if (indices.size(1) == 2) return backward_t<float, 2>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 3) return backward_t<float, 3>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 4) return backward_t<float, 4>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 5) return backward_t<float, 5>(grad_atomic_energies, nu1_basis, indices, multipliers);
            if (indices.size(1) == 6) return backward_t<float, 6>(grad_atomic_energies, nu1_basis, indices, multipliers);
            throw std::runtime_error("Polynomial order is too high.");
        } else {
            throw std::runtime_error("Unsupported dtype");
        }
    }
};


torch::Tensor polyeval(torch::Tensor nu1_basis, torch::Tensor indices, torch::Tensor multipliers) {
    return Polyeval::apply(nu1_basis, indices, multipliers);
}


TORCH_LIBRARY(polyeval_cc, m) {
    m.def("polyeval", &polyeval);
}

