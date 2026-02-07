#ifndef MPPI_KERNELS_CUH
#define MPPI_KERNELS_CUH

#include <cuda_runtime.h>
#include <type_traits>
#include "mppi_common.cuh"

namespace mppi {
namespace kernels {

// SFINAE helper to detect STATE_DIM and CONTROL_DIM
template <typename T, typename = void>
struct get_dims {
    static constexpr int STATE_DIM = -1;
    static constexpr int CONTROL_DIM = -1;
};

template <typename T>
struct get_dims<T, std::void_t<decltype(T::STATE_DIM), decltype(T::CONTROL_DIM)>> {
    static constexpr int STATE_DIM = T::STATE_DIM;
    static constexpr int CONTROL_DIM = T::CONTROL_DIM;
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

    // Use compile-time constants if available, otherwise fallback to local arrays
    constexpr int MAX_NX_DEFAULT = 32;
    constexpr int MAX_NU_DEFAULT = 12;

    constexpr int DYN_NX = get_dims<Dynamics>::STATE_DIM;
    constexpr int DYN_NU = get_dims<Dynamics>::CONTROL_DIM;

    // Determine array sizes
    constexpr int NX_ALLOC = (DYN_NX > 0) ? DYN_NX : MAX_NX_DEFAULT;
    constexpr int NU_ALLOC = (DYN_NU > 0) ? DYN_NU : MAX_NU_DEFAULT;

    // Determine loop bounds
    // If DYN_NX > 0, the loop bound is constant, enabling unrolling.
    // Otherwise, it uses runtime config.nx.
    int nx_loop = (DYN_NX > 0) ? DYN_NX : config.nx;
    int nu_loop = (DYN_NU > 0) ? DYN_NU : config.nu;

    float x[NX_ALLOC];
    float u[NU_ALLOC];
    float x_next[NX_ALLOC];

    // Copy initial state
    for(int i=0; i<nx_loop; ++i) {
        x[i] = initial_state[i];
    }
    
    float total_cost = 0.0f;
    
    for (int t = 0; t < config.horizon; ++t) {
        // Compute control u = u_nom[t] + noise[k, t]
        for(int i=0; i<nu_loop; ++i) {
            // noise index: k * (T * nu) + t * nu + i
            int noise_idx;
            if constexpr (DYN_NU > 0) {
                 noise_idx = k * (config.horizon * DYN_NU) + t * DYN_NU + i;
            } else {
                 noise_idx = k * (config.horizon * config.nu) + t * config.nu + i;
            }

            float n_val = noise[noise_idx];
            
            // u_nom index: t * nu + i
            int u_idx;
            if constexpr (DYN_NU > 0) {
                u_idx = t * DYN_NU + i;
            } else {
                u_idx = t * config.nu + i;
            }

            float u_val = u_nom[u_idx] + n_val;
            
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
