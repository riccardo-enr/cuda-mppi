#ifndef MPPI_KERNELS_CUH
#define MPPI_KERNELS_CUH

#include <cuda_runtime.h>

#include "mppi_common.cuh"

namespace mppi {
namespace kernels {

template <typename Dynamics, typename Cost>
__global__ void rollout_kernel(
    Dynamics dynamics, Cost cost, MPPIConfig config,
    const float* __restrict__ initial_state,
    const float* __restrict__ u_nom,      // Nominal control (T, nu)
    const float* __restrict__ noise,      // Noise (K, T, nu)
    const float* __restrict__ u_applied,  // Last applied control (nu)
    float* __restrict__ costs             // Output costs (K)
) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= config.num_samples) {
    return;
  }

  // Use compile-time constants if available, otherwise fallback to local arrays
  // assuming reasonable max dimensions or dynamic indexing if supported
  constexpr int MAX_NX = 32;
  constexpr int MAX_NU = 12;

  float x[MAX_NX];
  float u[MAX_NU];
  float u_prev[MAX_NU];
  float x_next[MAX_NX];

  // Copy initial state
  for (int i = 0; i < config.nx; ++i) {
    x[i] = initial_state[i];
  }

  // Initialize u_prev from last applied control
  for (int i = 0; i < config.nu; ++i) {
    u_prev[i] = u_applied[i];
  }

  float total_cost = 0.0f;

  for (int t = 0; t < config.horizon; ++t) {
    // Compute control: perturbed (most samples) or pure noise (exploration)
    const bool pure_noise =
        config.pure_noise_percentage > 0.0f &&
        k >= static_cast<int>((1.0f - config.pure_noise_percentage) *
                              config.num_samples);

    for (int i = 0; i < config.nu; ++i) {
      int noise_idx = k * (config.horizon * config.nu) + t * config.nu + i;
      float n_val = noise[noise_idx] * config.control_sigma[i];

      if (pure_noise) {
        u[i] = n_val;  // Zero-mean exploration
      } else {
        u[i] = u_nom[t * config.nu + i] + n_val;
      }
    }

    // Step dynamics
    dynamics.step(x, u, x_next, config.dt);

    // Compute cost (with rate-of-change via u_prev)
    total_cost += cost.compute(x, u, u_prev, t);

    // Likelihood ratio cost (importance sampling correction)
    if (config.alpha < 1.0f) {
      float lr_cost = 0.0f;
      for (int i = 0; i < config.nu; ++i) {
        float m = u_nom[t * config.nu + i];
        float s = config.control_sigma[i];
        lr_cost += m * (m - 2.0f * u[i]) / (s * s);
      }
      total_cost += 0.5f * config.lambda * (1.0f - config.alpha) * lr_cost;
    }

    // Update u_prev for next timestep
    for (int i = 0; i < config.nu; ++i) {
      u_prev[i] = u[i];
    }

    // Update state
    for (int i = 0; i < config.nx; ++i) {
      x[i] = x_next[i];
    }
  }

  // Terminal cost
  total_cost += cost.terminal_cost(x);

  costs[k] = total_cost;
}

}  // namespace kernels
}  // namespace mppi

#endif  // MPPI_KERNELS_CUH
