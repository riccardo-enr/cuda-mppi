#ifndef MPPI_KERNELS_CUH
#define MPPI_KERNELS_CUH

#include <cuda_runtime.h>
#include <type_traits>
#include "mppi_common.cuh"

namespace mppi {
namespace kernels {

// SFINAE helper to detect STATE_DIM and CONTROL_DIM
template <typename T, typename Enable = void>
struct DynamicsDims {
    static constexpr int STATE_DIM = 32;
    static constexpr int CONTROL_DIM = 12;
    static constexpr bool HAS_DIMS = false;
};

template <typename T>
struct DynamicsDims<T, std::void_t<decltype(T::STATE_DIM), decltype(T::CONTROL_DIM)>> {
    static constexpr int STATE_DIM = T::STATE_DIM;
    static constexpr int CONTROL_DIM = T::CONTROL_DIM;
    static constexpr bool HAS_DIMS = true;
};

template <typename Dynamics, typename Cost>
__global__ void rollout_kernel(
    Dynamics dynamics,
    Cost cost,
    MPPIConfig config,
    const float* __restrict__ initial_state,
    const float* __restrict__ u_nom,    // Nominal control (T, nu)
    const float* __restrict__ noise,    // Noise (K, T, nu)
    float* __restrict__ costs           // Output costs (K)
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= config.num_samples) return;

    // Use compile-time constants if available to reduce register pressure and enable unrolling
    constexpr int NX = DynamicsDims<Dynamics>::STATE_DIM;
    constexpr int NU = DynamicsDims<Dynamics>::CONTROL_DIM;
    constexpr bool HAS_DIMS = DynamicsDims<Dynamics>::HAS_DIMS;

    float x[NX];
    float u[NU];
    float x_next[NX];

    // Determine loop bounds: use constant if available, otherwise runtime config
    int nx_loop = HAS_DIMS ? NX : config.nx;
    int nu_loop = HAS_DIMS ? NU : config.nu;

    // Copy initial state
    #pragma unroll
    for(int i=0; i<nx_loop; ++i) {
        x[i] = initial_state[i];
    }
    
    float total_cost = 0.0f;
    
    for (int t = 0; t < config.horizon; ++t) {
        // Compute control u = u_nom[t] + noise[k, t]
        #pragma unroll
        for(int i=0; i<nu_loop; ++i) {
            // noise index: k * (T * nu) + t * nu + i
            int noise_idx = k * (config.horizon * config.nu) + t * config.nu + i;
            float n_val = noise[noise_idx];
            
            // u_nom index: t * nu + i
            float u_val = u_nom[t * config.nu + i] + n_val;
            
            // Apply scale
            u_val *= config.u_scale;
            
            // TODO: Apply bounds (u_min, u_max) if passed in config or accessible
            
            u[i] = u_val;
        }
        
        // Step dynamics
        dynamics.step(x, u, x_next, config.dt);
        
        // Compute cost
        total_cost += cost.compute(x, u, t);
        
        // Update state
        #pragma unroll
        for(int i=0; i<nx_loop; ++i) {
            x[i] = x_next[i];
        }
    }
    
    // Terminal cost
    total_cost += cost.terminal_cost(x);
    
    costs[k] = total_cost;
}

} // namespace kernels
} // namespace mppi

#endif // MPPI_KERNELS_CUH
