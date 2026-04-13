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
 * Scales raw $\mathcal{N}(0,1)$ velocity noise by $\sigma_i / \Delta t$ before
 * integration, so the resulting action perturbation has the intended magnitude:
 *
 * $$
 *   \mathbf{a}_k[t] = \mathbf{a}_{\text{nom}}[t]
 *     + \left(\mathbf{v}_{\text{nom}}[t]
 *       + \boldsymbol{\epsilon}_k[t] \odot \frac{\boldsymbol{\sigma}}{\Delta t}
 *     \right) \Delta t
 *   = \mathbf{a}_{\text{nom}}[t] + \mathbf{v}_{\text{nom}}[t]\,\Delta t
 *     + \boldsymbol{\epsilon}_k[t] \odot \boldsymbol{\sigma}
 * $$
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
 * @brief Recompute the nominal action sequence from the nominal velocity.
 *
 * Anchors at $\mathbf{a}_{\text{nom}}[0]$ (unchanged) and integrates forward:
 *
 * $$
 *   \mathbf{a}_{\text{nom}}[t] = \mathbf{a}_{\text{nom}}[0]
 *     + \sum_{s=0}^{t-1} \mathbf{v}_{\text{nom}}[s] \, \Delta t, \quad t \ge 1
 * $$
 *
 * Run with $n_u$ threads; each thread handles one control channel.
 *
 * @param a_nom  Nominal action sequence to update in-place $[T \times n_u]$.
 * @param v_nom  Nominal velocity sequence $[T \times n_u]$.
 * @param config MPPI configuration.
 */
__global__ void recompute_action_from_velocity_kernel(
  float * a_nom,
  const float * v_nom,
  MPPIConfig config
);

/**
 * @brief S-MPPI rollout kernel using pre-computed per-sample action sequences.
 *
 * Reads controls directly from `u_all[k,t,i]` without any sigma re-scaling.
 * The perturbed actions are already fully-formed by `integrate_actions_kernel`
 * (which applies the correct $\sigma / \Delta t$ velocity noise scaling).
 *
 * @tparam Dynamics  GPU-callable dynamics model.
 * @tparam Cost      GPU-callable cost function.
 * @param dynamics       Dynamics model.
 * @param cost           Cost function.
 * @param config         MPPI configuration.
 * @param initial_state  Current state $[n_x]$.
 * @param u_all          Pre-computed per-sample action sequences $[K \times T \times n_u]$.
 * @param u_applied      Last applied control $[n_u]$ (for rate cost at $t=0$).
 * @param costs          Per-sample total costs (output) $[K]$.
 */
template <typename Dynamics, typename Cost>
__global__ void smppi_rollout_kernel(
  Dynamics dynamics, Cost cost, MPPIConfig config,
  const float * __restrict__ initial_state,
  const float * __restrict__ u_all,
  const float * __restrict__ u_applied,
  float * __restrict__ costs)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= config.num_samples) return;

  constexpr int MAX_NX = 32;
  constexpr int MAX_NU = 12;

  float x[MAX_NX];
  float u[MAX_NU];
  float u_prev[MAX_NU];
  float x_next[MAX_NX];

  for (int i = 0; i < config.nx; ++i) x[i] = initial_state[i];
  for (int i = 0; i < config.nu; ++i) u_prev[i] = u_applied[i];

  float total_cost = 0.0f;

  for (int t = 0; t < config.horizon; ++t) {
    for (int i = 0; i < config.nu; ++i) {
      u[i] = u_all[k * config.horizon * config.nu + t * config.nu + i];
    }

    dynamics.step(x, u, x_next, config.dt);
    total_cost += cost.compute(x, u, u_prev, t);

    for (int i = 0; i < config.nu; ++i) u_prev[i] = u[i];
    for (int i = 0; i < config.nx; ++i) x[i] = x_next[i];
  }

  total_cost += cost.terminal_cost(x);
  costs[k] = total_cost;
}

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
    size_t noise_count_ = static_cast<size_t>(config.num_samples) * config.horizon * config.nu;
    noise_count_ += (noise_count_ & 1u);
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
    curandDestroyGenerator(gen_);
    softmax_.free();
  }

  /**
   * @brief Upload a reference control sequence for biased sampling.
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
    HANDLE_ERROR(cudaMemcpy(d_u_ref_, u_ref.data(), expected * sizeof(float),
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
   * 1. Sample velocity noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$
   * 2. Integrate to get perturbed actions: $\mathbf{a}_k = \mathbf{a}_{\text{nom}} + \mathbf{v}_{\text{nom}}\Delta t + \boldsymbol{\epsilon}_k \odot \boldsymbol{\sigma}$
   * 3. Rollout each sample using its pre-computed action sequence directly
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

    size_t noise_n = static_cast<size_t>(config_.num_samples) * config_.horizon * config_.nu;
    noise_n += (noise_n & 1u);
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

    // Rollout: read pre-computed perturbed actions directly (no sigma re-scaling)
    smppi_rollout_kernel << < grid, block >> > (
      dynamics_,
      cost_,
      config_,
      d_initial_state_,
      d_perturbed_actions_,
      d_u_applied_,
      d_costs_
      );
    HANDLE_ERROR(cudaGetLastError());

    // Add smoothness cost
    smoothness_cost_kernel << < grid, block >> > (
      d_perturbed_actions_,
      d_costs_,
      config_
      );
    HANDLE_ERROR(cudaGetLastError());

    /* Compute softmax weights on device (no PCIe round-trip). */
    softmax_.compute(d_costs_, d_weights_, config_.lambda, config_.num_samples);

    int num_params = config_.horizon * config_.nu;
    int blocks_upd = (num_params + 256 - 1) / 256;

    /* Update velocity in the same space as the N(0,1) noise (scaled by sigma),
       consistent with standard weighted_update_kernel semantics. */
    weighted_update_kernel << < blocks_upd, 256 >> > (
      d_u_vel_,
      d_noise_vel_,
      d_weights_,
      config_.num_samples,
      num_params,
      config_
      );
    HANDLE_ERROR(cudaGetLastError());

    /* Recompute action sequence from updated velocity; anchor at a_nom[0]. */
    recompute_action_from_velocity_kernel<<<1, config_.nu>>>(
      d_action_seq_,
      d_u_vel_,
      config_
      );
    HANDLE_ERROR(cudaGetLastError());
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

  /** @brief Get the current cost function (by value). */
  Cost cost() const { return cost_; }

  /** @brief Replace the cost function. */
  void set_cost(const Cost & c) { cost_ = c; }

  /**
   * @brief Set nominal control by broadcasting a single vector across the horizon.
   * @param u  Control vector $[n_u]$.
   */
  void set_nominal_control(const Eigen::VectorXf & u)
  {
    std::vector<float> flat(config_.horizon * config_.nu);
    for (int t = 0; t < config_.horizon; ++t) {
      for (int i = 0; i < config_.nu; ++i) {
        flat[t * config_.nu + i] = u[i];
      }
    }
    HANDLE_ERROR(cudaMemcpy(d_action_seq_, flat.data(),
        flat.size() * sizeof(float), cudaMemcpyHostToDevice));
  }

private:
  MPPIConfig config_;   ///< MPPI hyperparameters.
  Dynamics dynamics_;   ///< Dynamics model.
  Cost cost_;           ///< Cost function.

  float * d_u_vel_;              ///< Velocity sequence $[T \times n_u]$.
  float * d_action_seq_;         ///< Integrated action sequence $[T \times n_u]$.
  float * d_noise_vel_;          ///< Velocity noise $[K \times T \times n_u]$.
  float * d_perturbed_actions_;  ///< Per-sample perturbed actions $[K \times T \times n_u]$.

  float * d_u_ref_;          ///< Reference bias sequence $[T \times n_u]$.

  float * d_costs_;          ///< Per-sample costs $[K]$.
  float * d_initial_state_;  ///< Current state $[n_x]$.
  float * d_weights_;        ///< Softmax weights $[K]$.
  float * d_u_applied_;      ///< Last applied control $[n_u]$.
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

      /* Scale noise by sigma/dt so that after integration by dt the action
         perturbation has magnitude sigma -- matching standard MPPI exploration. */
      float scaled_noise = noise_vel[idx_sample] * config.control_sigma[i] / config.dt;
      float vel = u_vel_nom[idx_base] + scaled_noise;

      perturbed_actions[idx_sample] = action_seq_nom[idx_base] + vel * config.dt;
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

__global__ void recompute_action_from_velocity_kernel(
  float * a_nom,
  const float * v_nom,
  MPPIConfig config
)
{
  /* One thread per control channel. Loop over horizon to stay serial within
     each channel so the cumulative sum stays numerically stable. */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= config.nu) return;

  float a = a_nom[i];  /* anchor: a_nom[0,i] is kept as-is */
  for (int t = 1; t < config.horizon; ++t) {
    a += v_nom[(t - 1) * config.nu + i] * config.dt;
    a_nom[t * config.nu + i] = a;
  }
}

}  // namespace mppi

#endif  // SMPPI_CONTROLLER_CUH
