/**
 * @file i_mppi.cuh
 * @brief Informative MPPI (I-MPPI) controller with biased sampling.
 *
 * Extends the base `MPPIController` to support **biased importance sampling**:
 * a fraction $\alpha$ of the $K$ noise samples are shifted so that the
 * resulting controls are centred around an informative reference trajectory
 * $\mathbf{u}_{\text{ref}}$ instead of the current nominal
 * $\mathbf{u}_{\text{nom}}$.
 *
 * ## Sampling distribution
 *
 * The mixture sampling distribution is:
 *
 * $$
 *   Q(\mathbf{u}) = (1 - \alpha)\,\mathcal{N}(\mathbf{u}_{\text{nom}},
 * \boldsymbol{\Sigma})
 *                 + \alpha\,\mathcal{N}(\mathbf{u}_{\text{ref}},
 * \boldsymbol{\Sigma})
 * $$
 *
 * This is implemented by shifting the noise of biased samples:
 * $\boldsymbol{\epsilon}'_k = \boldsymbol{\epsilon}_k +
 * (\mathbf{u}_{\text{ref}} - \mathbf{u}_{\text{nom}})$, so that
 * $\mathbf{u}_{\text{nom}} + \boldsymbol{\epsilon}'_k$ is centred at
 * $\mathbf{u}_{\text{ref}}$.
 *
 * @tparam Dynamics  Dynamics model (GPU-callable).
 * @tparam Cost      Cost function (GPU-callable).
 */

#ifndef IMPPI_CONTROLLER_CUH
#define IMPPI_CONTROLLER_CUH

#include "mppi/controllers/mppi.cuh"

namespace mppi {

/**
 * @brief Informative MPPI controller.
 *
 * Inherits from `MPPIController` and overrides `compute()` to inject
 * biased noise samples toward $\mathbf{u}_{\text{ref}}$ before the
 * standard rollout and weight computation.
 *
 * ### Usage
 *
 * ```cpp
 * IMPPIController<MyDynamics, MyCost> ctrl(config, dynamics, cost);
 * ctrl.set_reference_trajectory(u_ref_flat);  // from informative planner
 * ctrl.compute(current_state);
 * auto action = ctrl.get_action();
 * ctrl.shift();
 * ```
 *
 * @tparam Dynamics  Dynamics model (GPU-callable).
 * @tparam Cost      Cost function (GPU-callable).
 */
template <typename Dynamics, typename Cost>
class IMPPIController : public MPPIController<Dynamics, Cost> {
 public:
  /**
   * @brief Construct the I-MPPI controller.
   *
   * Allocates the additional device buffer for $\mathbf{u}_{\text{ref}}$.
   *
   * @param config    MPPI config (uses `alpha` for bias fraction).
   * @param dynamics  Dynamics model instance.
   * @param cost      Cost function instance.
   */
  IMPPIController(const MPPIConfig &config, const Dynamics &dynamics,
                  const Cost &cost)
      : MPPIController<Dynamics, Cost>(config, dynamics, cost) {
    HANDLE_ERROR(
        cudaMalloc(&d_u_ref_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(
        cudaMemset(d_u_ref_, 0, config.horizon * config.nu * sizeof(float)));
  }

  /** @brief Destructor. Frees the reference trajectory buffer. */
  ~IMPPIController() { cudaFree(d_u_ref_); }

  /**
   * @brief Upload an informative reference trajectory to the device.
   *
   * @param u_ref_flat  Flattened reference controls $\in \mathbb{R}^{T \cdot
   * n_u}$.
   */
  void set_reference_trajectory(const Eigen::VectorXf &u_ref_flat) {
    if (u_ref_flat.size() != this->config_.horizon * this->config_.nu) {
      std::cerr << "Error: Reference trajectory size mismatch!" << std::endl;
      return;
    }
    HANDLE_ERROR(
        cudaMemcpy(d_u_ref_, u_ref_flat.data(),
                   this->config_.horizon * this->config_.nu * sizeof(float),
                   cudaMemcpyHostToDevice));
  }

  /**
   * @brief Update cost parameters (goal position and weight).
   *
   * @param goal          Goal position (NED).
   * @param lambda_goal   Goal attraction weight.
   */
  void update_cost_params(float3 goal, float lambda_goal) {
    this->cost_.goal = goal;
    this->cost_.lambda_goal = lambda_goal;
  }

  /**
   * @brief Run the I-MPPI optimization loop (biased sampling variant).
   *
   * Steps:
   * 1. Copy state to device
   * 2. Sample $K$ standard normal noise vectors
   * 3. Shift the last $\lfloor \alpha K \rfloor$ samples toward
   * $\mathbf{u}_{\text{ref}}$
   * 4. Rollout all $K$ trajectories
   * 5. Compute softmax importance weights
   * 6. Update $\mathbf{u}_{\text{nom}}$ via weighted noise sum
   *
   * @param state  Current state $\mathbf{x} \in \mathbb{R}^{n_x}$.
   */
  void compute(const Eigen::VectorXf &state) {
    HANDLE_ERROR(cudaMemcpy(this->d_initial_state_, state.data(),
                            this->config_.nx * sizeof(float),
                            cudaMemcpyHostToDevice));

    HANDLE_CURAND_ERROR(curandGenerateNormal(
        this->gen_, this->d_noise_,
        this->config_.num_samples * this->config_.horizon * this->config_.nu,
        0.0f, 1.0f));

    // Apply bias to alpha fraction of samples
    int num_biased = (int)(this->config_.num_samples * this->config_.alpha);
    int start_biased_idx = this->config_.num_samples - num_biased;

    if (num_biased > 0) {
      dim3 block(256);
      dim3 grid((this->config_.num_samples + block.x - 1) / block.x);

      kernels::apply_bias_kernel<<<grid, block>>>(
          this->d_noise_, this->d_u_nom_, d_u_ref_, this->config_.num_samples,
          this->config_.horizon, this->config_.nu, start_biased_idx);
      HANDLE_ERROR(cudaGetLastError());
    }

    // Rollout (standard kernel — biased noise makes u centred at u_ref)
    dim3 block(256);
    dim3 grid((this->config_.num_samples + block.x - 1) / block.x);

    kernels::imppi_rollout_kernel<<<grid, block>>>(
        this->dynamics_, this->cost_, this->config_, this->d_initial_state_,
        this->d_u_nom_, this->d_noise_, this->d_u_applied_, this->d_costs_);
    HANDLE_ERROR(cudaGetLastError());

    /* Compute softmax weights on device (no PCIe round-trip). */
    this->softmax_.compute(this->d_costs_, this->d_weights_,
                           this->config_.lambda, this->config_.num_samples);

    int num_params = this->config_.horizon * this->config_.nu;
    int threads = 256;
    int blocks = (num_params + threads - 1) / threads;

    weighted_update_kernel<<<blocks, threads>>>(
        this->d_u_nom_, this->d_noise_, this->d_weights_,
        this->config_.num_samples, num_params, this->config_);
    HANDLE_ERROR(cudaGetLastError());
  }

 private:
  float *d_u_ref_;  ///< Informative reference trajectory $[T \times n_u]$
                    ///< (device).
};

}  // namespace mppi

#endif  // IMPPI_CONTROLLER_CUH
