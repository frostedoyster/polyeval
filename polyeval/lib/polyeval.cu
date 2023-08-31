#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define FULL_MASK 0xffffffff

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr,
                                    std::size_t *space = nullptr) noexcept
{
    const std::uintptr_t inptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t end = inptr + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - inptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(inptr);
}

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

#define NTHREADS_FOR_BASIS 32
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> nu1_basis,   // [natoms, n_nu1]
    const torch::PackedTensorAccessor32<int16_t, 2, torch::RestrictPtrTraits> indices,      // [nbasis, polynomial_order]
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> multipliers, // [nbasis]
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output)            // [natoms]
{
    extern __shared__ char buffer[];
    void *sptr = buffer;
    size_t space = 0;

    /* SHARED BUFFERS */
    scalar_t *buffer_nu1 = shared_array<scalar_t>(nu1_basis.size(1), sptr, &space);
    scalar_t *monomial_buffer = shared_array<scalar_t>(blockDim.x, sptr, &space);
    scalar_t *basis_buffer = shared_array<scalar_t>(NTHREADS_FOR_BASIS, sptr, &space);

    int32_t nbasis = multipliers.size(0);

    // load all of nu1_basis into shared memory
    for (int32_t i = threadIdx.x; i < nu1_basis.size(0); i += blockDim.x)
    {
        buffer_nu1[i] = nu1_basis[blockIdx.x][i];
    }

    __syncthreads();

    scalar_t atom_energy = 0.0;

    int nbasis_loops = find_integer_divisor(nbasis, NTHREADS_FOR_BASIS); // 32

    for (int basis_loop = 0; basis_loop < nbasis_loops; basis_loop++)
    {
        int nthreadx = 4;
        int tidx = threadIdx.x % nthreadx;
        int tidy = threadIdx.x / nthreadx;
        int nthready = blockDim.x / nthreadx;

        int basis = basis_loop * NTHREADS_FOR_BASIS + tidy;

        for (int i_monomial = tidx; i_monomial < indices.size(1); i_monomial += nthreadx)
        {
            scalar_t val = 0.0;

            if (basis < nbasis)
            {
                int16_t idx = indices[basis][i_monomial];
                val = buffer_nu1[idx];
            }

            monomial_buffer[tidy * nthreadx + tidx] = val;
        }

        __syncthreads();

        scalar_t temp = 0.0;

        if (basis < nbasis)
        {
            temp = multipliers[basis];
        }

        // now multiply-reduce the monomial terms, and write to a new buffer for futher reduction
        if (tidx == 0)
        {
            for (int i_monomial = 0; i_monomial < indices.size(1); i_monomial++)
            {
                temp *= monomial_buffer[tidy * nthreadx + i_monomial];
            }

            basis_buffer[tidy] = temp;
        }

        __syncthreads();

        //
        nthreadx = 32;
        tidx = threadIdx.x % nthreadx;
        tidy = threadIdx.x / nthreadx;
        nthready = blockDim.x / nthreadx;

        // now use the first warp to reduce into atom_energy;
        if (tidy == 0)
        {
            scalar_t val = basis_buffer[tidx];
            for (int offset = nthreadx / 2; offset > 0; offset /= 2)
            {
                val += __shfl_down_sync(FULL_MASK, val, offset);
            }

            atom_energy += val;
        }
        __syncthreads();
    }

    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        output[blockIdx.x] = atom_energy;
    }
}

template <typename scalar_t, uint8_t n_monomials>
__global__ void forward2_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> nu1_basis,   // [natoms, n_nu1]
    const torch::PackedTensorAccessor32<int16_t, 2, torch::RestrictPtrTraits> indices,      // [nbasis, polynomial_order]
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> multipliers, // [nbasis]
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output)            // [natoms]
{
    extern __shared__ char buffer[];

    int32_t natoms = nu1_basis.size(0);
    int32_t nnu1 = nu1_basis.size(1);
    int32_t nbasis = multipliers.size(0);

    void *sptr = buffer;
    size_t space = 0;

    /* SHARED BUFFERS */
    scalar_t *buffer_nu1 = shared_array<scalar_t>(blockDim.y * nnu1, sptr, &space);

    int32_t atom_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (atom_id > natoms)
    {
        return;
    }

    // load all of nu1_basis into shared memory
    for (int32_t i = threadIdx.x; i < nnu1; i += blockDim.x)
    {
        buffer_nu1[threadIdx.y * nnu1 + i] = nu1_basis[atom_id][i];
    }

    __syncthreads();

    scalar_t atom_energy = 0.0; // 1 per y
    scalar_t c = 0.0;           // kahans summation...
    output[atom_id] = 0.0;

    for (int32_t basis = threadIdx.x; basis < nbasis; basis += blockDim.x)
    {
        scalar_t tmp = 1.0;

#pragma unroll
        for (uint8_t i_monomial = 0; i_monomial < n_monomials; i_monomial++)
        {
            int16_t idx = indices[i_monomial][basis];

            scalar_t val = buffer_nu1[threadIdx.y * nnu1 + idx];

            tmp *= val;
        }

        // kahans summation for F32
        scalar_t y = tmp * multipliers[basis] - c;
        scalar_t t = atom_energy + y;
        c = (t - atom_energy) - y;
        atom_energy = t;
    }

    for (int32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        atom_energy += __shfl_down_sync(FULL_MASK, atom_energy, offset);
    }

    /*// if using nthreadx > 32, then we need to add % 32 == 0 in shared mem
    if (blockDim.x > WARP_SIZE && threadIdx.x % WARP_SIZE == 0)
    {
        buffer_output[threadIdx.x / WARP_SIZE] = atom_energy;
    } */

    if (threadIdx.x % WARP_SIZE == 0)
    {
        atomicAdd(&output[atom_id], atom_energy);
    }
}

template <uint8_t n_monomials>
torch::Tensor forward_gpu(torch::Tensor nu1_basis,
                          torch::Tensor indices,
                          torch::Tensor multipliers,
                          int64_t nthreadx,
                          int64_t nthready,
                          int64_t nthreadz)
{

    int natoms = nu1_basis.size(0);
    int nnu1 = nu1_basis.size(1);

    torch::Tensor output = torch::empty({natoms},
                                        torch::TensorOptions()
                                            .dtype(nu1_basis.dtype())
                                            .device(nu1_basis.device()));

    int nb = find_integer_divisor(natoms, nthready);

    dim3 block_dim(nb);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        nu1_basis.type(), "forward_gpu", ([&]
                                          {
                                              void *sptr = nullptr;
                                              size_t space = 0;

                                              shared_array<scalar_t>(nthready * nnu1, sptr, &space);

                                              forward2_kernel<scalar_t, n_monomials><<<block_dim, grid_dim, space>>>(
                                                  nu1_basis.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                  indices.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(),
                                                  multipliers.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                  output.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()); }

                                          ));

    cudaDeviceSynchronize();

    return output;
}

class PolyEvalCUDA : public Function<PolyEvalCUDA>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor nu1_basis,
        torch::Tensor indices,
        torch::Tensor multipliers,
        int64_t nthreadx,
        int64_t nthready)
    {
        torch::Tensor result;
        if (indices.size(0) == 2)
        {
            result = forward_gpu<2>(nu1_basis, indices, multipliers, nthreadx, nthready, 1);
        }
        else if (indices.size(0) == 3)
        {
            result = forward_gpu<3>(nu1_basis, indices, multipliers, nthreadx, nthready, 1);
        }
        else if (indices.size(0) == 4)
        {
            result = forward_gpu<4>(nu1_basis, indices, multipliers, nthreadx, nthready, 1);
        }
        else if (indices.size(0) == 5)
        {
            result = forward_gpu<5>(nu1_basis, indices, multipliers, nthreadx, nthready, 1);
        }
        else if (indices.size(0) == 6)
        {
            result = forward_gpu<6>(nu1_basis, indices, multipliers, nthreadx, nthready, 1);
        }

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {

        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor polyeval(torch::Tensor nu1_basis, torch::Tensor indices, torch::Tensor multipliers, int64_t nthreadx, int64_t nthready)
{
    return PolyEvalCUDA::apply(nu1_basis, indices, multipliers, nthreadx, nthready);
}

TORCH_LIBRARY(polyeval_cu, m)
{
    m.def("polyeval", &polyeval);
}
