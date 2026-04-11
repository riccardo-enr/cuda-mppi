/**
 * @file mppi.cuh
 * @brief GPU-accelerated Model Predictive Path Integral (MPPI) controller.
 *
 * Implements the MPPI stochastic optimal control algorithm using CUDA.
 * The controller samples random control perturbations, evaluates their
 * cost via parallel GPU rollouts, and updates the nominal trajectory
 * using importance-weighted averaging.
 *
 * ## Algorithm
 *
 * Given a nominal control sequence $\mathbf{u}_{0:T-1}$, the controller:
 * 1. Samples $K$ perturbations $\boldsymbol{\epsilon}_k \sim \mathcal{N}(0,
 * \boldsymbol{\Sigma})$
 * 2. Rolls out each perturbed trajectory and computes its cost $S_k$
 * 3. Computes importance weights $w_k = \frac{\exp(-S_k / \lambda)}{\sum_j
 * \exp(-S_j / \lambda)}$
 * 4. Updates the nominal sequence: $\mathbf{u} \leftarrow \mathbf{u} + \alpha
 * \sum_k w_k \boldsymbol{\epsilon}_k$
 *
 * @tparam Dynamics  Callable dynamics model with `step()` method
 * (GPU-compatible).
 * @tparam Cost      Callable cost function with `running_cost()` /
 * `terminal_cost()`.
 *
 * @see Williams et al., "Information Theoretic MPC for Model-Based
 * Reinforcement Learning", ICRA 2017.
 */

#ifndef MPPI_CONTROLLER_CUH
#define MPPI_CONTROLLER_CUH

#include <cuda_runtime.h>
#include <curand.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "mppi/core/kernels.cuh"
#include "mppi/core/mppi_common.cuh"
#include "mppi/core/softmax.cuh"
#include "mppi/utils/cuda_utils.cuh"

namespace mppi {

/**
 * @brief CUDA kernel for additive weighted update of the nominal control
 * sequence.
 *
 * Computes the single-iteration MPPI update:
 *
 * $$
 *   \mathbf{u}[i] \leftarrow \mathbf{u}[i] + \alpha \sum_{k=0}^{K-1} w_k \,
 * \epsilon_k[i] \, \sigma_{i \bmod n_u}
 * $$
 *
 * where $\epsilon_k$ is standard normal noise and $\sigma$ is the per-dimension
 * control standard deviation.
 *
 * @param[in,out] u_nom        Nominal control sequence on device $[T \times
 * n_u]$.
 * @param[in]     noise        Sampled $\mathcal{N}(0,1)$ noise on device $[K
 * \times T \times n_u]$.
 * @param[in]     weights      Softmax importance weights on device $[K]$.
 * @param[in]     K            Number of samples (rollouts).
 * @param[in]     total_params Total control parameters $(T \times n_u)$.
 * @param[in]     config       MPPI configuration (passed by value to GPU
 * registers).
 */
__global__ void weighted_update_kernel(float* u_nom, const float* noise,
                                       const float* weights, int K,
                                       int total_params,  // T * nu
                                       MPPIConfig config);

/**
 * @brief CUDA kernel for weighted mean replacement of the nominal control
 * sequence.
 *
 * Computes the multi-iteration mean replacement update:
 *
 * $$
 *   \mathbf{u}[i] \leftarrow \sum_{k=0}^{K-1} w_k \left( \mathbf{u}[i] +
 * \epsilon_k[i] \, \sigma_{i \bmod n_u} \right)
 * $$
 *
 * Unlike the additive update, this **replaces** the nominal with the weighted
 * mean of all sampled trajectories. Suitable for multi-iteration refinement
 * with decaying $\sigma$.
 *
 * @param[in,out] u_nom        Nominal control sequence on device $[T \times
 * n_u]$.
 * @param[in]     noise        Sampled $\mathcal{N}(0,1)$ noise on device $[K
 * \times T \times n_u]$.
 * @param[in]     weights      Softmax importance weights on device $[K]$.
 * @param[in]     K            Number of samples (rollouts).
 * @param[in]     total_params Total control parameters $(T \times n_u)$.
 * @param[in]     config       MPPI configuration (passed by value to GPU
 * registers).
 */
__global__ void weighted_mean_kernel(float* u_nom, const float* noise,
                                     const float* weights, int K,
                                     int total_params,  // T * nu
                                     MPPIConfig config);

/**
 * @brief CUDA kernel that shifts the nominal control sequence forward by one
 * timestep.
 *
 * Copies $\mathbf{u}[t+1]$ into $\mathbf{u}[t]$ for all timesteps.
 * The last timestep retains its previous value (warm-start) rather than being
 * zeroed.
 *
 * @param[in,out] u_nom Nominal control sequence on device $[T \times n_u]$.
 * @param[in]     T     Prediction horizon length.
 * @param[in]     nu    Control dimensionality $n_u$.
 */
__global__ void shift_kernel(float* u_nom, int T, int nu);

/**
 * @brief GPU-accelerated MPPI stochastic optimal controller.
 *
 * Templated on user-provided `Dynamics` and `Cost` types that must be
 * GPU-callable (i.e., `__device__`-compatible). The controller manages all
 * device memory for the nominal control sequence, noise samples, rollout
 * costs, and importance weights.
 *
 * ### Usage
 *
 * ```cpp
 * MPPIController<MyDynamics, MyCost> ctrl(config, dynamics, cost);
 * ctrl.compute(current_state);       // run MPPI optimization
 * auto action = ctrl.get_action();   // retrieve first optimal control
 * ctrl.shift();                      // shift horizon for next timestep
 * ```
 *
 * @tparam Dynamics  Dynamics model providing `step(state, control) ->
 * next_state`.
 * @tparam Cost      Cost function providing `running_cost(state, control)` and
 *                   `terminal_cost(state)`.
 */
template <typename Dynamics, typename Cost>
class MPPIController {
 public:
  /**
   * @brief Construct the MPPI controller and allocate all GPU memory.
   *
   * Allocates device buffers for the nominal control sequence
   * $\mathbf{u} \in \mathbb{R}^{T \times n_u}$, noise samples, costs,
   * weights, and initializes the CuRAND pseudo-random generator.
   *
   * @param config   MPPI hyperparameters (horizon, samples, $\lambda$,
   * $\sigma$, etc.).
   * @param dynamics Dynamics model instance (copied to host member).
   * @param cost     Cost function instance (copied to host member).
   */
  MPPIController(const MPPIConfig& config, const Dynamics& dynamics,
                 const Cost& cost)
      : config_(config), dynamics_(dynamics), cost_(cost) {
    // Allocate device memory
    HANDLE_ERROR(
        cudaMalloc(&d_u_nom_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_noise_, config.num_samples * config.horizon *
                                           config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_costs_, config.num_samples * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_initial_state_, config.nx * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_weights_, config.num_samples * sizeof(float)));

    // Initialize u_nom to zero
    HANDLE_ERROR(
        cudaMemset(d_u_nom_, 0, config.horizon * config.nu * sizeof(float)));

    // Reference bias buffer (for split sampling)
    HANDLE_ERROR(
        cudaMalloc(&d_u_ref_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(
        cudaMemset(d_u_ref_, 0, config.horizon * config.nu * sizeof(float)));

    // Last applied control (for rate-of-change cost)
    HANDLE_ERROR(cudaMalloc(&d_u_applied_, config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_applied_, 0, config.nu * sizeof(float)));

    // Setup CuRAND
    HANDLE_CURAND_ERROR(
        curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));

    softmax_.allocate(config.num_samples);
  }

  /**
   * @brief Destructor. Frees all device memory and destroys the CuRAND
   * generator.
   */
  ~MPPIController() {
    cudaFree(d_u_nom_);
    cudaFree(d_u_ref_);
    cudaFree(d_costs_);
    cudaFree(d_initial_state_);
    cudaFree(d_weights_);
    cudaFree(d_u_applied_);
    curandDestroyGenerator(gen_);
    softmax_.free();
  }

  /** @brief Replace the cost function instance. */
  void set_cost(const Cost& cost) { cost_ = cost; }
  /** @brief Mutable access to the cost function. */
  Cost& cost() { return cost_; }
  /** @brief Const access to the cost function. */
  const Cost& cost() const { return cost_; }
  /** @brief Replace the dynamics model instance. */
  void set_dynamics(const Dynamics& dynamics) { dynamics_ = dynamics; }
  /** @brief Mutable access to the dynamics model. */
  Dynamics& dynamics() { return dynamics_; }
  /** @brief Raw device pointer to the nominal control sequence $[T \times
   * n_u]$. */
  float* get_u_nom_ptr() { return d_u_nom_; }

  /**
   * @brief Upload the last applied control to device memory.
   *
   * Used by the cost function to penalize control rate-of-change
   * $\|\mathbf{u}_0 - \mathbf{u}_{\text{prev}}\|^2$.
   *
   * @param u Last applied control vector $\in \mathbb{R}^{n_u}$.
   */
  void set_applied_control(const Eigen::VectorXf& u) {
    HANDLE_ERROR(cudaMemcpy(d_u_applied_, u.data(), config_.nu * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  /**
   * @brief Upload a reference control sequence for biased sampling.
   *
   * When set, a fraction `alpha` of noise samples will be shifted toward
   * this reference, implementing split sampling (warm-start + bias).
   *
   * @param u_ref Flattened reference controls $\in \mathbb{R}^{T \cdot n_u}$.
   */
  void set_reference_sequence(const std::vector<float>& u_ref) {
    int expected = config_.horizon * config_.nu;
    if (static_cast<int>(u_ref.size()) != expected) {
      return;
    }
    HANDLE_ERROR(cudaMemcpy(d_u_ref_, u_ref.data(), expected * sizeof(float),
                            cudaMemcpyHostToDevice));
    has_ref_bias_ = true;
  }

  /**
   * @brief Clear the reference bias. All samples will explore around u_nom.
   */
  void clear_reference_sequence() { has_ref_bias_ = false; }

  /**
   * @brief Run the full MPPI optimization loop.
   *
   * Executes `num_iters` refinement iterations. Each iteration:
   * 1. Samples $K$ noise vectors $\boldsymbol{\epsilon} \sim \mathcal{N}(0,
   * \mathbf{I})$
   * 2. Launches `rollout_kernel` to evaluate all $K$ trajectories in parallel
   * 3. Computes softmax importance weights on the host:
   *    $w_k = \frac{\exp(-(S_k - S_{\min}) / \lambda)}{\sum_j \exp(-(S_j -
   * S_{\min}) / \lambda)}$
   * 4. Updates $\mathbf{u}$ via `weighted_update_kernel` (single iter) or
   *    `weighted_mean_kernel` (multi-iter with decaying $\sigma$)
   *
   * After calling `compute()`, use `get_action()` to retrieve the first
   * optimal control.
   *
   * @param state Current state vector $\mathbf{x} \in \mathbb{R}^{n_x}$.
   */
  void compute(const Eigen::VectorXf& state) {
    // Copy state to device
    HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(),
                            config_.nx * sizeof(float),
                            cudaMemcpyHostToDevice));

    const int num_iters = config_.num_iters > 0 ? config_.num_iters : 1;

    for (int iter = 0; iter < num_iters; ++iter) {
      // Build per-iteration config with decayed sigma
      MPPIConfig iter_config = config_;
      if (config_.std_dev_decay < 1.0f && iter > 0) {
        float decay = powf(config_.std_dev_decay, static_cast<float>(iter));
        for (int i = 0; i < config_.nu; ++i) {
          iter_config.control_sigma[i] = config_.control_sigma[i] * decay;
        }
      }

      // Sample N(0,1) noise
      HANDLE_CURAND_ERROR(curandGenerateNormal(
          gen_, d_noise_, config_.num_samples * config_.horizon * config_.nu,
          0.0f, 1.0f));

      // Apply reference bias to alpha fraction of samples (split sampling)
      if (has_ref_bias_ && iter_config.alpha > 0.0f) {
        int num_biased =
            static_cast<int>(iter_config.num_samples * iter_config.alpha);
        int start_biased_idx = iter_config.num_samples - num_biased;
        if (num_biased > 0) {
          dim3 bias_block(256);
          dim3 bias_grid((iter_config.num_samples + bias_block.x - 1) /
                         bias_block.x);
          kernels::apply_bias_kernel<<<bias_grid, bias_block>>>(
              d_noise_, d_u_nom_, d_u_ref_, iter_config.num_samples,
              iter_config.horizon, iter_config.nu, start_biased_idx);
          HANDLE_ERROR(cudaGetLastError());
        }
      }

      // Launch Rollout Kernel
      dim3 block(256);
      dim3 grid((config_.num_samples + block.x - 1) / block.x);

      kernels::rollout_kernel<<<grid, block>>>(
          dynamics_, cost_, iter_config, d_initial_state_, d_u_nom_, d_noise_,
          d_u_applied_, d_costs_);
      HANDLE_ERROR(cudaGetLastError());

      /* Compute softmax weights on device (no PCIe round-trip). */
      softmax_.compute(d_costs_, d_weights_, iter_config.lambda,
                       config_.num_samples);

      int num_params = config_.horizon * config_.nu;
      int threads = 256;
      int blocks = (num_params + threads - 1) / threads;

      if (num_iters > 1) {
        // Multi-iteration: replace u_nom with weighted mean of samples
        weighted_mean_kernel<<<blocks, threads>>>(
            d_u_nom_, d_noise_, d_weights_, config_.num_samples, num_params,
            iter_config);
      } else {
        // Single iteration: additive update (original behavior)
        weighted_update_kernel<<<blocks, threads>>>(
            d_u_nom_, d_noise_, d_weights_, config_.num_samples, num_params,
            iter_config);
      }
      HANDLE_ERROR(cudaGetLastError());
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  /**
   * @brief Shift the nominal control sequence forward by one timestep.
   *
   * After applying the first control, call this to advance the horizon:
   * $\mathbf{u}[t] \leftarrow \mathbf{u}[t+1]$ for $t = 0, \ldots, T-2$.
   * The last timestep is held constant (warm-start).
   */
  void shift() {
    const int total_floats = config_.horizon * config_.nu;
    const int threads = 256;
    const int blocks = (total_floats + threads - 1) / threads;
    shift_kernel<<<blocks, threads>>>(d_u_nom_, config_.horizon, config_.nu);
  }

  /**
   * @brief Set every timestep of the nominal sequence to the same control
   * value.
   *
   * Useful for initializing the trajectory with a constant control
   * (e.g., hover thrust).
   *
   * @param u Control vector $\in \mathbb{R}^{n_u}$ to broadcast across all $T$
   * steps.
   */
  void set_nominal_control(const Eigen::VectorXf& u) {
    // Set each timestep to the same control value
    for (int t = 0; t < config_.horizon; ++t) {
      HANDLE_ERROR(cudaMemcpy(d_u_nom_ + t * config_.nu, u.data(),
                              config_.nu * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
  }

  /**
   * @brief Retrieve the first optimal control action from the nominal sequence.
   *
   * Copies $\mathbf{u}[0] \in \mathbb{R}^{n_u}$ from device to host.
   *
   * @return First control action of the optimized horizon.
   */
  Eigen::VectorXf get_action() {
    // Return first action from d_u_nom_
    Eigen::VectorXf action(config_.nu);
    HANDLE_ERROR(cudaMemcpy(action.data(), d_u_nom_, config_.nu * sizeof(float),
                            cudaMemcpyDeviceToHost));
    return action;
  }

 protected:
  MPPIConfig config_;  ///< MPPI hyperparameters.
  Dynamics dynamics_;  ///< Dynamics model instance.
  Cost cost_;          ///< Cost function instance.

  /// @name Device memory buffers
  /// @{
  float* d_u_nom_;          ///< Nominal control sequence $[T \times n_u]$.
  float* d_u_ref_;          ///< Reference bias sequence $[T \times n_u]$.
  float* d_noise_;          ///< Sampled noise $[K \times T \times n_u]$.
  float* d_costs_;          ///< Per-sample rollout costs $[K]$.
  float* d_initial_state_;  ///< Current state $[n_x]$.
  float* d_weights_;        ///< Softmax importance weights $[K]$.
  float* d_u_applied_;      ///< Last applied control $[n_u]$ (for rate cost).
  bool has_ref_bias_ = false;  ///< Whether reference bias is active.
  SoftmaxWeights softmax_;  ///< GPU-side softmax helper (CUB reductions).
  /// @}

  curandGenerator_t gen_;  ///< CuRAND pseudo-random number generator.
};

// Helper Kernels

__global__ void weighted_update_kernel(float* u_nom, const float* noise,
                                       const float* weights, int K,
                                       int total_params,  // T * nu
                                       MPPIConfig config) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_params) {
    return;
  }

  // Determine which control dimension this index corresponds to.
  // config is passed by value so control_sigma lives in GPU registers.
  float sigma = config.control_sigma[idx % config.nu];

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    // noise is N(0,1); rollout used noise*sigma as the actual perturbation,
    // so the weighted update must also scale by sigma to stay consistent.
    sum += weights[k] * noise[k * total_params + idx] * sigma;
  }

  u_nom[idx] += config.learning_rate * sum;
}

// Weighted mean update: u_nom = Σ w_k * (u_nom + noise_k * sigma)
// For multi-iteration refinement where we replace the mean each iteration.
__global__ void weighted_mean_kernel(float* u_nom, const float* noise,
                                     const float* weights, int K,
                                     int total_params, MPPIConfig config) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_params) {
    return;
  }

  float sigma = config.control_sigma[idx % config.nu];
  float current = u_nom[idx];

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    float sample = current + noise[k * total_params + idx] * sigma;
    sum += weights[k] * sample;
  }

  u_nom[idx] = sum;
}

__global__ void shift_kernel(float* u_nom, int T, int nu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = T * nu;
  if (idx >= total) {
    return;
  }

  if (idx < (T - 1) * nu) {
    u_nom[idx] = u_nom[idx + nu];
  }
  // else: hold last value (warm-start with previous tail instead of zeroing)
}

}  // namespace mppi

#endif  // MPPI_CONTROLLER_CUH
