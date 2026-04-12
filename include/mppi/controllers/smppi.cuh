/**
 * @file smppi.cuh
 * @brief Smooth MPPI (S-MPPI) controller with velocity-space optimisation.
 *
 * Instead of optimising the control sequence directly, S-MPPI optimises
 * a **velocity** (rate-of-change) sequence $\mathbf{v}_{0:T-1}$ and
 * integrates it to obtain the action sequence:
 *
 * $$
 *   \mathbf{a}[t] = \mathbf{a}_{\text{nom}}[t] + \mathbf{v}[t] \, \Delta t
 * $$
 *
 * An additional smoothness cost penalises consecutive action differences:
 *
 * $$
 *   S_{\text{smooth}} = w_s \sum_{t=0}^{T-2} \|\mathbf{a}[t+1] - \mathbf{a}[t]\|^2
 * $$
 *
 * This produces inherently smooth trajectories without requiring explicit
 * constraints or kernel interpolation.
 *
 * @tparam Dynamics  Dynamics model (GPU-callable).
 * @tparam Cost      Cost function (GPU-callable).
 */

#ifndef SMPPI_CONTROLLER_CUH
#define SMPPI_CONTROLLER_CUH

#include "mppi.cuh"

namespace mppi {

/**
 * @brief CUDA kernel to integrate velocity perturbations into action sequences.
 *
 * For each sample $k$, computes:
 * $\mathbf{a}_k[t] = \mathbf{a}_{\text{nom}}[t] + (\mathbf{v}_{\text{nom}}[t] + \boldsymbol{\epsilon}_k[t]) \, \Delta t$
 *
 * One thread per sample; loops over $T$ and $n_u$.
 *
 * @param action_seq_nom  Nominal action sequence $[T \times n_u]$.
 * @param u_vel_nom       Nominal velocity sequence $[T \times n_u]$.
 * @param noise_vel       Velocity noise $[K \times T \times n_u]$.
 * @param perturbed_actions  Output perturbed actions $[K \times T \times n_u]$.
 * @param config          MPPI configuration.
 */
__global__ void integrate_actions_kernel(
  const float * action_seq_nom,
  const float * u_vel_nom,
  const float * noise_vel,
  float * perturbed_actions,
  MPPIConfig config
);

/**
 * @brief CUDA kernel to integrate velocity into the nominal action sequence.
 *
 * Applies $\mathbf{a}[i] \mathrel{+}= \mathbf{v}[i] \, \Delta t$ element-wise.
 * One thread per element of the flattened $[T \times n_u]$ array.
 *
 * @param action_seq  Action sequence to update in-place $[T \times n_u]$.
 * @param u_vel       Velocity sequence $[T \times n_u]$.
 * @param config      MPPI configuration (provides `dt`).
 */
__global__ void integrate_single_action_kernel(
  float * action_seq,
  const float * u_vel,
  MPPIConfig config
);

/**
 * @brief CUDA kernel to compute the action-sequence smoothness cost.
 *
 * For each sample $k$, accumulates:
 *
 * $$
 *   S_k \mathrel{+}= w_s \sum_{t=0}^{T-2} \sum_i (\mathbf{a}_k[t+1, i] - \mathbf{a}_k[t, i])^2
 * $$
 *
 * @param perturbed_actions  Per-sample action sequences $[K \times T \times n_u]$.
 * @param costs              Per-sample costs to augment $[K]$.
 * @param config             MPPI configuration (provides `w_action_seq_cost`).
 */
__global__ void smoothness_cost_kernel(
  const float * perturbed_actions,
  float * costs,
  MPPIConfig config
);

/**
 * @brief Smooth MPPI controller.
 *
 * Optimises in velocity space and integrates to produce smooth action
 * sequences. Adds a smoothness penalty on top of the standard rollout cost.
 *
 * ### Usage
 *
 * ```cpp
 * SMPPIController<MyDynamics, MyCost> ctrl(config, dynamics, cost);
 * ctrl.compute(current_state);
 * auto action = ctrl.get_action();
 * ctrl.shift();
 * ```
 *
 * @tparam Dynamics  Dynamics model (GPU-callable).
 * @tparam Cost      Cost function (GPU-callable).
 */
template<typename Dynamics, typename Cost>
class SMPPIController {
public:
  /**
   * @brief Construct the S-MPPI controller and allocate device memory.
   *
   * @param config    MPPI config (uses `w_action_seq_cost` for smoothness weight).
   * @param dynamics  Dynamics model instance.
   * @param cost      Cost function instance.
   */
  SMPPIController(const MPPIConfig & config, const Dynamics & dynamics, const Cost & cost)
  : config_(config), dynamics_(dynamics), cost_(cost)
  {

    HANDLE_ERROR(cudaMalloc(&d_u_vel_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_action_seq_, config.horizon * config.nu * sizeof(float)));
    /* curandGenerateNormal requires an even element count. */
    int noise_count_ = config.num_samples * config.horizon * config.nu;
    noise_count_ += (noise_count_ & 1);
    HANDLE_ERROR(cudaMalloc(&d_noise_vel_, noise_count_ * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_perturbed_actions_,
        config.num_samples * config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_costs_, config.num_samples * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_initial_state_, config.nx * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_weights_, config.num_samples * sizeof(float)));

    HANDLE_ERROR(cudaMemset(d_u_vel_, 0, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_action_seq_, 0, config.horizon * config.nu * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&d_u_applied_, config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_applied_, 0, config.nu * sizeof(float)));

    // Zero buffer used as u_nom placeholder in rollout
    HANDLE_ERROR(cudaMalloc(&d_zeros_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_zeros_, 0, config.horizon * config.nu * sizeof(float)));

    // Reference bias buffer (for split sampling)
    HANDLE_ERROR(cudaMalloc(&d_u_ref_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_ref_, 0, config.horizon * config.nu * sizeof(float)));

    HANDLE_CURAND_ERROR(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));

    softmax_.allocate(config.num_samples);
  }

  /** @brief Destructor. Frees all device buffers. */
  ~SMPPIController()
  {
    cudaFree(d_u_vel_);
    cudaFree(d_action_seq_);
    cudaFree(d_noise_vel_);
    cudaFree(d_perturbed_actions_);
    cudaFree(d_u_ref_);
    cudaFree(d_costs_);
    cudaFree(d_initial_state_);
    cudaFree(d_weights_);
    cudaFree(d_u_applied_);
    cudaFree(d_zeros_);
    curandDestroyGenerator(gen_);
    softmax_.free();
  }

  /**
   * @brief Upload a reference control sequence for biased sampling.
   *
   * The reference is divided by control_sigma before uploading so that
   * biased samples center at the actual u_ref after sigma scaling in rollout.
   *
   * @param u_ref Flattened reference controls $\in \mathbb{R}^{T \cdot n_u}$.
   * @param bias_alpha Fraction of samples to bias (separate from config_.alpha
   *                   to avoid LR cost issues with zeros u_nom).
   */
  void set_reference_sequence(const std::vector<float>& u_ref, float bias_alpha) {
    int expected = config_.horizon * config_.nu;
    if (static_cast<int>(u_ref.size()) != expected) {
      return;
    }
    // Normalize by sigma: biased samples will be d_perturbed_actions_ which
    // goes through rollout as u = 0 + perturbed * sigma, so we need
    // u_ref_normalized = u_ref / sigma for the bias to produce correct values
    std::vector<float> u_ref_norm(expected);
    for (int t = 0; t < config_.horizon; ++t) {
      for (int i = 0; i < config_.nu; ++i) {
        float sigma = config_.control_sigma[i];
        u_ref_norm[t * config_.nu + i] = u_ref[t * config_.nu + i] / sigma;
      }
    }
    HANDLE_ERROR(cudaMemcpy(d_u_ref_, u_ref_norm.data(), expected * sizeof(float),
                            cudaMemcpyHostToDevice));
    has_ref_bias_ = true;
    bias_alpha_ = bias_alpha;
  }

  /**
   * @brief Clear the reference bias. All samples will explore freely.
   */
  void clear_reference_sequence() {
    has_ref_bias_ = false;
    bias_alpha_ = 0.0f;
  }

  /**
   * @brief Run the S-MPPI optimization loop.
   *
   * 1. Sample velocity noise
   * 2. Integrate to get perturbed action sequences
   * 3. Rollout using perturbed actions as "noise" (with zero nominal)
   * 4. Add smoothness cost
   * 5. Compute softmax weights, update velocity sequence
   * 6. Integrate updated velocity into action sequence
   *
   * @param state  Current state $\mathbf{x} \in \mathbb{R}^{n_x}$.
   */
  void compute(const Eigen::VectorXf & state)
  {
    HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(), config_.nx * sizeof(float),
        cudaMemcpyHostToDevice));

    int noise_n = config_.num_samples * config_.horizon * config_.nu;
    noise_n += (noise_n & 1);
    HANDLE_CURAND_ERROR(curandGenerateNormal(gen_, d_noise_vel_, noise_n, 0.0f, 1.0f));

    dim3 block(256);
    dim3 grid((config_.num_samples + block.x - 1) / block.x);

    integrate_actions_kernel << < grid, block >> > (
      d_action_seq_,
      d_u_vel_,
      d_noise_vel_,
      d_perturbed_actions_,
      config_
      );
    HANDLE_ERROR(cudaGetLastError());

    // Apply reference bias to alpha fraction of samples (split sampling)
    // Uses d_action_seq_ as u_nom (the nominal integrated actions) and
    // d_u_ref_ which has been sigma-normalized in set_reference_sequence()
    if (has_ref_bias_ && bias_alpha_ > 0.0f) {
      int num_biased = static_cast<int>(config_.num_samples * bias_alpha_);
      int start_biased_idx = config_.num_samples - num_biased;
      if (num_biased > 0) {
        dim3 bias_block(256);
        dim3 bias_grid((config_.num_samples + bias_block.x - 1) / bias_block.x);
        kernels::apply_bias_kernel<<<bias_grid, bias_block>>>(
            d_perturbed_actions_, d_action_seq_, d_u_ref_, config_.num_samples,
            config_.horizon, config_.nu, start_biased_idx);
        HANDLE_ERROR(cudaGetLastError());
      }
    }

    // Rollout with perturbed actions (u_nom=0, noise=perturbed_actions)
    kernels::rollout_kernel << < grid, block >> > (
      dynamics_,
      cost_,
      config_,
      d_initial_state_,
      d_zeros_,
      d_perturbed_actions_,
      d_u_applied_,
      d_costs_
      );

    // Add smoothness cost
    smoothness_cost_kernel << < grid, block >> > (
      d_perturbed_actions_,
      d_costs_,
      config_
      );

    /* Compute softmax weights on device (no PCIe round-trip). */
    softmax_.compute(d_costs_, d_weights_, config_.lambda, config_.num_samples);

    int num_params = config_.horizon * config_.nu;
    int blocks_upd = (num_params + 256 - 1) / 256;

    weighted_update_kernel << < blocks_upd, 256 >> > (
      d_u_vel_,
      d_noise_vel_,
      d_weights_,
      config_.num_samples,
      num_params,
      config_
      );

    // Integrate updated velocity into action sequence
    const int interp_threads = (config_.horizon * config_.nu + 255) / 256;
    integrate_single_action_kernel<<<interp_threads, 256>>>(
      d_action_seq_,
      d_u_vel_,
      config_
      );
  }

  /**
   * @brief Shift both velocity and action sequences forward by one timestep.
   */
  void shift()
  {
    const int total_floats = config_.horizon * config_.nu;
    const int threads = 256;
    const int blocks = (total_floats + threads - 1) / threads;

    shift_kernel<<<blocks, threads>>>(d_u_vel_, config_.horizon, config_.nu);
    shift_kernel<<<blocks, threads>>>(d_action_seq_, config_.horizon, config_.nu);
  }

  /** @brief Upload the last applied control (for rate cost). */
  void set_applied_control(const Eigen::VectorXf& u) {
    HANDLE_ERROR(cudaMemcpy(d_u_applied_, u.data(),
                            config_.nu * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  /**
   * @brief Retrieve the first optimal control action.
   * @return First action from the integrated action sequence.
   */
  Eigen::VectorXf get_action()
  {
    Eigen::VectorXf action(config_.nu);
    HANDLE_ERROR(cudaMemcpy(action.data(), d_action_seq_, config_.nu * sizeof(float),
        cudaMemcpyDeviceToHost));
    return action;
  }

private:
  MPPIConfig config_;   ///< MPPI hyperparameters.
  Dynamics dynamics_;   ///< Dynamics model.
  Cost cost_;           ///< Cost function.

  float * d_u_vel_;              ///< Velocity sequence $[T \times n_u]$.
  float * d_action_seq_;         ///< Integrated action sequence $[T \times n_u]$.
  float * d_noise_vel_;          ///< Velocity noise $[K \times T \times n_u]$.
  float * d_perturbed_actions_;  ///< Per-sample perturbed actions $[K \times T \times n_u]$.

  float * d_u_ref_;          ///< Reference bias sequence $[T \times n_u]$ (sigma-normalized).

  float * d_costs_;          ///< Per-sample costs $[K]$.
  float * d_initial_state_;  ///< Current state $[n_x]$.
  float * d_weights_;        ///< Softmax weights $[K]$.
  float * d_u_applied_;      ///< Last applied control $[n_u]$.
  float * d_zeros_;          ///< Zero buffer (rollout placeholder) $[T \times n_u]$.
  bool has_ref_bias_ = false; ///< Whether reference bias is active.
  float bias_alpha_ = 0.0f;  ///< Fraction of samples to bias (separate from config_.alpha).
  SoftmaxWeights softmax_;   ///< GPU-side softmax helper (CUB reductions).

  curandGenerator_t gen_;  ///< CuRAND generator.
};

// ===========================================================================
// Kernel Implementations
// ===========================================================================

__global__ void integrate_actions_kernel(
  const float * action_seq_nom,
  const float * u_vel_nom,
  const float * noise_vel,
  float * perturbed_actions,
  MPPIConfig config
)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= config.num_samples) {return;}

  for (int t = 0; t < config.horizon; ++t) {
    for (int i = 0; i < config.nu; ++i) {
      int idx_base = t * config.nu + i;
      int idx_sample = k * (config.horizon * config.nu) + idx_base;

      float vel = u_vel_nom[idx_base] + noise_vel[idx_sample];

      float act = action_seq_nom[idx_base] + vel * config.dt;

      perturbed_actions[idx_sample] = act;
    }
  }
}

__global__ void integrate_single_action_kernel(
  float * action_seq,
  const float * u_vel,
  MPPIConfig config
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= config.horizon * config.nu) { return; }

  action_seq[idx] += u_vel[idx] * config.dt;
}

__global__ void smoothness_cost_kernel(
  const float * perturbed_actions,
  float * costs,
  MPPIConfig config
)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= config.num_samples) {return;}

  float cost = 0.0f;
  for (int t = 0; t < config.horizon - 1; ++t) {
    for (int i = 0; i < config.nu; ++i) {
      int idx1 = k * (config.horizon * config.nu) + t * config.nu + i;
      int idx2 = k * (config.horizon * config.nu) + (t + 1) * config.nu + i;

      float diff = (perturbed_actions[idx2] - perturbed_actions[idx1]) * config.u_scale;
      cost += diff * diff;
    }
  }
  costs[k] += cost * config.w_action_seq_cost;
}

}  // namespace mppi

#endif  // SMPPI_CONTROLLER_CUH
