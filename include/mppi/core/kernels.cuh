/**
 * @file kernels.cuh
 * @brief Core CUDA rollout kernel for MPPI trajectory sampling.
 *
 * Each GPU thread simulates one complete trajectory of length $T$, applying
 * perturbed controls and accumulating the running + terminal cost. This is
 * the computational bottleneck of MPPI — all $K$ rollouts execute in parallel.
 */

#ifndef MPPI_KERNELS_CUH
#define MPPI_KERNELS_CUH

#include <cuda_runtime.h>

#include "mppi_common.cuh"

namespace mppi {
namespace kernels {

/**
 * @brief Pure MPPI parallel trajectory rollout kernel.
 *
 * Thread $k$ simulates a full $T$-step trajectory starting from
 * `initial_state` and accumulates the total cost into `costs[k]`.
 *
 * ## Control perturbation
 *
 * For most samples the applied control at timestep $t$ is:
 *
 * $$
 *   \mathbf{u}_k[t] = \mathbf{u}_{\text{nom}}[t] + \boldsymbol{\epsilon}_k[t] \odot \boldsymbol{\sigma}
 * $$
 *
 * A fraction `pure_noise_percentage` of samples use zero-mean noise
 * ($\mathbf{u}_k[t] = \boldsymbol{\epsilon}_k[t] \odot \boldsymbol{\sigma}$) for exploration.
 *
 * @tparam Dynamics  Model with `__device__ void step(x, u, x_next, dt)`.
 * @tparam Cost      Cost with `__device__ float compute(x, u, u_prev, t)`
 *                   and `__device__ float terminal_cost(x)`.
 *
 * @param[in]  dynamics      Dynamics model (passed by value to registers).
 * @param[in]  cost          Cost function (passed by value to registers).
 * @param[in]  config        MPPI configuration.
 * @param[in]  initial_state Current state $\mathbf{x}_0 \in \mathbb{R}^{n_x}$.
 * @param[in]  u_nom         Nominal control sequence $[T \times n_u]$.
 * @param[in]  noise         $\mathcal{N}(0,1)$ samples $[K \times T \times n_u]$.
 * @param[in]  u_applied     Last applied control $[n_u]$ (for rate cost at $t=0$).
 * @param[out] costs         Per-sample total cost $[K]$.
 */
template <typename Dynamics, typename Cost>
__global__ void rollout_kernel(
    Dynamics dynamics, Cost cost, MPPIConfig config,
    const float* __restrict__ initial_state,
    const float* __restrict__ u_nom,
    const float* __restrict__ noise,
    const float* __restrict__ u_applied,
    float* __restrict__ costs) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= config.num_samples) {
    return;
  }

  constexpr int MAX_NX = 32;
  constexpr int MAX_NU = 12;

  float x[MAX_NX];
  float u[MAX_NU];
  float u_prev[MAX_NU];
  float x_next[MAX_NX];

  for (int i = 0; i < config.nx; ++i) {
    x[i] = initial_state[i];
  }
  for (int i = 0; i < config.nu; ++i) {
    u_prev[i] = u_applied[i];
  }

  float total_cost = 0.0f;

  for (int t = 0; t < config.horizon; ++t) {
    const bool pure_noise =
        config.pure_noise_percentage > 0.0f &&
        k >= static_cast<int>((1.0f - config.pure_noise_percentage) *
                              config.num_samples);

    for (int i = 0; i < config.nu; ++i) {
      int noise_idx = k * (config.horizon * config.nu) + t * config.nu + i;
      float n_val = noise[noise_idx] * config.control_sigma[i];

      if (pure_noise) {
        u[i] = n_val;
      } else {
        u[i] = u_nom[t * config.nu + i] + n_val;
      }
    }

    dynamics.step(x, u, x_next, config.dt);
    total_cost += cost.compute(x, u, u_prev, t);

    for (int i = 0; i < config.nu; ++i) {
      u_prev[i] = u[i];
    }
    for (int i = 0; i < config.nx; ++i) {
      x[i] = x_next[i];
    }
  }

  total_cost += cost.terminal_cost(x);
  costs[k] = total_cost;
}

/**
 * @brief I-MPPI rollout kernel with importance-sampling correction.
 *
 * Identical to `rollout_kernel` but adds a likelihood-ratio cost term
 * to correct for the biased sampling distribution used by I-MPPI:
 *
 * $$
 *   S_k \mathrel{+}= \frac{\lambda (1 - \alpha)}{2} \sum_{t,i}
 *     \frac{m_{t,i} (m_{t,i} - 2 u_{k,t,i})}{\sigma_i^2}
 * $$
 *
 * where $m_{t,i} = u_{\text{nom}}[t,i]$ and $\alpha$ is the fraction
 * of samples biased toward the informative reference trajectory.
 *
 * This kernel should only be used by `IMPPIController`.
 */
template <typename Dynamics, typename Cost>
__global__ void imppi_rollout_kernel(
    Dynamics dynamics, Cost cost, MPPIConfig config,
    const float* __restrict__ initial_state,
    const float* __restrict__ u_nom,
    const float* __restrict__ noise,
    const float* __restrict__ u_applied,
    float* __restrict__ costs) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= config.num_samples) {
    return;
  }

  constexpr int MAX_NX = 32;
  constexpr int MAX_NU = 12;

  float x[MAX_NX];
  float u[MAX_NU];
  float u_prev[MAX_NU];
  float x_next[MAX_NX];

  for (int i = 0; i < config.nx; ++i) {
    x[i] = initial_state[i];
  }
  for (int i = 0; i < config.nu; ++i) {
    u_prev[i] = u_applied[i];
  }

  float total_cost = 0.0f;

  for (int t = 0; t < config.horizon; ++t) {
    const bool pure_noise =
        config.pure_noise_percentage > 0.0f &&
        k >= static_cast<int>((1.0f - config.pure_noise_percentage) *
                              config.num_samples);

    for (int i = 0; i < config.nu; ++i) {
      int noise_idx = k * (config.horizon * config.nu) + t * config.nu + i;
      float n_val = noise[noise_idx] * config.control_sigma[i];

      if (pure_noise) {
        u[i] = n_val;
      } else {
        u[i] = u_nom[t * config.nu + i] + n_val;
      }
    }

    dynamics.step(x, u, x_next, config.dt);
    total_cost += cost.compute(x, u, u_prev, t);

    // Importance-sampling likelihood-ratio correction for biased sampling.
    if (config.alpha > 0.0f && config.alpha < 1.0f) {
      float lr_cost = 0.0f;
      for (int i = 0; i < config.nu; ++i) {
        float m = u_nom[t * config.nu + i];
        float s = config.control_sigma[i];
        lr_cost += m * (m - 2.0f * u[i]) / (s * s);
      }
      total_cost += 0.5f * config.lambda * (1.0f - config.alpha) * lr_cost;
    }

    for (int i = 0; i < config.nu; ++i) {
      u_prev[i] = u[i];
    }
    for (int i = 0; i < config.nx; ++i) {
      x[i] = x_next[i];
    }
  }

  total_cost += cost.terminal_cost(x);
  costs[k] = total_cost;
}

/**
 * @brief CUDA kernel to shift noise samples toward a reference trajectory.
 *
 * For samples with index $k \geq$ `start_biased_idx`, applies:
 *
 * $$
 *   \boldsymbol{\epsilon}_k[t, i] \mathrel{+}= u_{\text{ref}}[t, i] - u_{\text{nom}}[t, i]
 * $$
 *
 * @param[in,out] noise             Noise buffer $[K \times T \times n_u]$.
 * @param[in]     u_nom             Current nominal sequence $[T \times n_u]$.
 * @param[in]     u_ref             Reference sequence $[T \times n_u]$.
 * @param[in]     num_samples       Total number of samples $K$.
 * @param[in]     horizon           Prediction horizon $T$.
 * @param[in]     nu                Control dimension $n_u$.
 * @param[in]     start_biased_idx  First sample index to bias.
 */
__global__ inline void apply_bias_kernel(
    float* noise,
    const float* u_nom,
    const float* u_ref,
    int num_samples,
    int horizon,
    int nu,
    int start_biased_idx) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_samples || k < start_biased_idx) {
    return;
  }

  for (int t = 0; t < horizon; ++t) {
    for (int i = 0; i < nu; ++i) {
      int idx = k * (horizon * nu) + t * nu + i;
      int u_idx = t * nu + i;
      noise[idx] += u_ref[u_idx] - u_nom[u_idx];
    }
  }
}

}  // namespace kernels
}  // namespace mppi

#endif  // MPPI_KERNELS_CUH
