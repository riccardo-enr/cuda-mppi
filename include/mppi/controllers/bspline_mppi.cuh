/**
 * @file bspline_mppi.cuh
 * @brief B-Spline MPPI controller with spline-space sampling.
 *
 * Instead of sampling per-timestep Gaussian noise, this controller
 * samples noise in the low-dimensional B-spline control-point space
 * and expands via De Boor evaluation to produce smooth control sequences.
 *
 * ## Sampling
 *
 * The nominal trajectory is represented by $n$ control points
 * $\mathbf{P}_i \in \mathbb{R}^{n_u}$. At each iteration:
 *
 * 1. Sample $\delta\mathbf{P}_i^{(k)} \sim \mathcal{N}(0, \boldsymbol{\Sigma}_P)$
 * 2. Expand $\mathbf{P}_i + \delta\mathbf{P}_i^{(k)}$ via cubic B-spline
 *    → per-timestep controls $\mathbf{u}_k(t)$
 * 3. Roll out dynamics and cost
 * 4. Update control points: $\mathbf{P}_i \leftarrow \mathbf{P}_i +
 *    \text{lr} \sum_k w_k \, \delta\mathbf{P}_i^{(k)} \sigma_{P,i}$
 *
 * The B-spline's $C^2$ continuity guarantees smooth acceleration and
 * angular acceleration profiles, reducing dynamically infeasible rollouts.
 *
 * ## I-MPPI biased sampling
 *
 * A fraction $\alpha$ of samples can be biased toward a reference
 * control-point trajectory $\mathbf{P}_{\text{ref}}$ (e.g., from FSMI-RLE
 * B-spline optimizer), analogous to the I-MPPI mechanism.
 *
 * @tparam Dynamics  GPU-callable dynamics model.
 * @tparam Cost      GPU-callable cost function.
 */

#ifndef BSPLINE_MPPI_CONTROLLER_CUH
#define BSPLINE_MPPI_CONTROLLER_CUH

#include <cuda_runtime.h>
#include <curand.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "mppi/core/kernels.cuh"
#include "mppi/core/mppi_common.cuh"
#include "mppi/core/bspline.cuh"
#include "mppi/utils/cuda_utils.cuh"

namespace mppi {

// ============================================================================
// Kernels for B-Spline MPPI
// ============================================================================

/**
 * @brief Build perturbed control-point sets from nominal + noise.
 *
 * For each sample k, control point j, dimension i:
 *   cp_out[k,j,i] = cp_nom[j,i] + noise[k,j,i] * sigma_cp[i]
 *
 * @param[in]  cp_nom     Nominal CPs [n_cp × nu].
 * @param[in]  noise      N(0,1) noise [K × n_cp × nu].
 * @param[out] cp_out     Perturbed CPs [K × n_cp × nu].
 * @param[in]  sigma_cp   Per-dimension CP noise std dev [nu] (max 12).
 * @param[in]  K          Number of samples.
 * @param[in]  n_cp       Number of control points.
 * @param[in]  nu         Control dimension.
 */
__global__ void build_cp_samples_kernel(
    const float* __restrict__ cp_nom,
    const float* __restrict__ noise,
    float* __restrict__ cp_out,
    const float* __restrict__ sigma_cp,
    int K, int n_cp, int nu)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * n_cp * nu;
  if (tid >= total) return;

  int rem = tid % (n_cp * nu);
  int i = rem % nu;

  int nom_idx = rem;  // j * nu + i within a single sample
  cp_out[tid] = cp_nom[nom_idx] + noise[tid] * sigma_cp[i];
}

/**
 * @brief Bias a fraction of CP noise samples toward a reference CP trajectory.
 *
 * For samples k >= start_biased_idx, shifts noise so that
 * cp_nom + noise * sigma centres at cp_ref instead of cp_nom:
 *   noise[k,j,i] += (cp_ref[j,i] - cp_nom[j,i]) / sigma_cp[i]
 *
 * @param[in,out] noise             CP noise [K × n_cp × nu].
 * @param[in]     cp_nom            Nominal CPs [n_cp × nu].
 * @param[in]     cp_ref            Reference CPs [n_cp × nu].
 * @param[in]     sigma_cp          Per-dim std dev [nu].
 * @param[in]     K                 Number of samples.
 * @param[in]     n_cp              Number of control points.
 * @param[in]     nu                Control dimension.
 * @param[in]     start_biased_idx  First sample index to bias.
 */
__global__ void apply_cp_bias_kernel(
    float* __restrict__ noise,
    const float* __restrict__ cp_nom,
    const float* __restrict__ cp_ref,
    const float* __restrict__ sigma_cp,
    int K, int n_cp, int nu,
    int start_biased_idx)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K || k < start_biased_idx) return;

  for (int j = 0; j < n_cp; ++j) {
    for (int i = 0; i < nu; ++i) {
      int idx = k * n_cp * nu + j * nu + i;
      int nom_idx = j * nu + i;
      float s = sigma_cp[i];
      if (s > 1e-8f) {
        noise[idx] += (cp_ref[nom_idx] - cp_nom[nom_idx]) / s;
      }
    }
  }
}

/**
 * @brief Rollout kernel using pre-computed per-sample control sequences.
 *
 * Like `rollout_kernel` but reads controls directly from `u_all[k,t,i]`
 * instead of computing u_nom + noise * sigma. No importance sampling
 * correction is applied (the correction is not needed when sampling in
 * a different space).
 *
 * @tparam Dynamics  GPU-callable dynamics model.
 * @tparam Cost      GPU-callable cost function.
 */
template <typename Dynamics, typename Cost>
__global__ void rollout_bspline_kernel(
    Dynamics dynamics, Cost cost, MPPIConfig config,
    const float* __restrict__ initial_state,
    const float* __restrict__ u_all,
    const float* __restrict__ u_applied,
    float* __restrict__ costs)
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
    // Read pre-computed control for this sample and timestep
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
 * @brief Weighted update of control points from CP-space noise.
 *
 * cp_nom[idx] += lr * Σ_k w_k * noise[k * total_cp_params + idx] * sigma_cp[idx % nu]
 *
 * @param[in,out] cp_nom      Nominal CPs [n_cp × nu].
 * @param[in]     cp_noise    CP noise [K × n_cp × nu].
 * @param[in]     weights     Softmax weights [K].
 * @param[in]     sigma_cp    Per-dim std dev [nu].
 * @param[in]     K           Number of samples.
 * @param[in]     total_params  n_cp × nu.
 * @param[in]     nu          Control dimension.
 * @param[in]     learning_rate  Update step size.
 */
__global__ void weighted_update_cp_kernel(
    float* __restrict__ cp_nom,
    const float* __restrict__ cp_noise,
    const float* __restrict__ weights,
    const float* __restrict__ sigma_cp,
    int K, int total_params, int nu,
    float learning_rate)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_params) return;

  float sigma = sigma_cp[idx % nu];
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += weights[k] * cp_noise[k * total_params + idx] * sigma;
  }

  cp_nom[idx] += learning_rate * sum;
}

/**
 * @brief Shift control points forward: remove first, duplicate last.
 *
 * After applying one timestep, the B-spline should shift forward in time.
 * This shifts CPs by one position and holds the last CP constant.
 *
 * @param[in,out] cp_nom  Control points [n_cp × nu].
 * @param[in]     n_cp    Number of control points.
 * @param[in]     nu      Control dimension.
 */
__global__ void shift_cp_kernel(float* cp_nom, int n_cp, int nu)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_cp * nu;
  if (idx >= total) return;

  if (idx < (n_cp - 1) * nu) {
    cp_nom[idx] = cp_nom[idx + nu];
  }
  // else: hold last CP (warm-start)
}

// ============================================================================
// B-Spline MPPI Controller
// ============================================================================

/**
 * @brief B-Spline MPPI controller with spline-space sampling.
 *
 * @tparam Dynamics  GPU-callable dynamics model.
 * @tparam Cost      GPU-callable cost function.
 */
template<typename Dynamics, typename Cost>
class BSplineMPPIController
{
public:
  /**
   * @brief Construct the B-Spline MPPI controller.
   *
   * @param config     MPPI hyperparameters.
   * @param dynamics   Dynamics model instance.
   * @param cost       Cost function instance.
   * @param bspline    B-spline configuration (n_cp, knot_span).
   */
  BSplineMPPIController(const MPPIConfig& config, const Dynamics& dynamics,
                        const Cost& cost, const BSplineConfig& bspline)
    : config_(config), dynamics_(dynamics), cost_(cost), bspline_(bspline)
  {
    const int cp_total = bspline_.n_cp * config_.nu;
    const int u_total = config_.horizon * config_.nu;

    // Nominal control points
    HANDLE_ERROR(cudaMalloc(&d_cp_nom_, cp_total * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_cp_nom_, 0, cp_total * sizeof(float)));

    // CP noise [K × n_cp × nu]
    HANDLE_ERROR(cudaMalloc(&d_cp_noise_,
        config_.num_samples * cp_total * sizeof(float)));

    // Perturbed CPs [K × n_cp × nu]
    HANDLE_ERROR(cudaMalloc(&d_cp_samples_,
        config_.num_samples * cp_total * sizeof(float)));

    // Expanded controls [K × H × nu]
    HANDLE_ERROR(cudaMalloc(&d_u_expanded_,
        config_.num_samples * u_total * sizeof(float)));

    // Expanded nominal [H × nu]
    HANDLE_ERROR(cudaMalloc(&d_u_nom_, u_total * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_nom_, 0, u_total * sizeof(float)));

    // Reference CPs for biased sampling [n_cp × nu]
    HANDLE_ERROR(cudaMalloc(&d_cp_ref_, cp_total * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_cp_ref_, 0, cp_total * sizeof(float)));

    // Per-dimension CP noise sigma [nu]
    HANDLE_ERROR(cudaMalloc(&d_sigma_cp_, config_.nu * sizeof(float)));
    // Default: use control_sigma from config
    HANDLE_ERROR(cudaMemcpy(d_sigma_cp_, config_.control_sigma,
        config_.nu * sizeof(float), cudaMemcpyHostToDevice));

    // Costs, weights, state, applied control
    HANDLE_ERROR(cudaMalloc(&d_costs_, config_.num_samples * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_weights_, config_.num_samples * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_initial_state_, config_.nx * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_u_applied_, config_.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_applied_, 0, config_.nu * sizeof(float)));

    // CuRAND
    HANDLE_CURAND_ERROR(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 42ULL));
  }

  ~BSplineMPPIController()
  {
    cudaFree(d_cp_nom_);
    cudaFree(d_cp_noise_);
    cudaFree(d_cp_samples_);
    cudaFree(d_u_expanded_);
    cudaFree(d_u_nom_);
    cudaFree(d_cp_ref_);
    cudaFree(d_sigma_cp_);
    cudaFree(d_costs_);
    cudaFree(d_weights_);
    cudaFree(d_initial_state_);
    cudaFree(d_u_applied_);
    curandDestroyGenerator(gen_);
  }

  // --- Accessors ---

  void set_cost(const Cost& cost) { cost_ = cost; }
  Cost& cost() { return cost_; }
  void set_dynamics(const Dynamics& dynamics) { dynamics_ = dynamics; }

  /** @brief Device pointer to nominal control points [n_cp × nu]. */
  float* get_cp_nom_ptr() { return d_cp_nom_; }

  /** @brief Device pointer to expanded nominal controls [H × nu]. */
  float* get_u_nom_ptr() { return d_u_nom_; }

  /**
   * @brief Set CP-space noise standard deviations.
   *
   * @param sigma  Per-dimension sigma [nu].
   */
  void set_cp_sigma(const std::vector<float>& sigma)
  {
    int n = std::min(static_cast<int>(sigma.size()), config_.nu);
    HANDLE_ERROR(cudaMemcpy(d_sigma_cp_, sigma.data(),
        n * sizeof(float), cudaMemcpyHostToDevice));
  }

  /**
   * @brief Upload a reference CP trajectory for biased sampling.
   *
   * @param cp_ref  Flattened reference CPs [n_cp × nu].
   */
  void set_reference_cp(const std::vector<float>& cp_ref)
  {
    int expected = bspline_.n_cp * config_.nu;
    if (static_cast<int>(cp_ref.size()) != expected) return;
    HANDLE_ERROR(cudaMemcpy(d_cp_ref_, cp_ref.data(),
        expected * sizeof(float), cudaMemcpyHostToDevice));
    has_cp_ref_ = true;
  }

  void clear_reference_cp() { has_cp_ref_ = false; }

  /**
   * @brief Set the nominal control points from host data.
   *
   * @param cp_nom  Flattened nominal CPs [n_cp × nu].
   */
  void set_nominal_cp(const std::vector<float>& cp_nom)
  {
    int expected = bspline_.n_cp * config_.nu;
    if (static_cast<int>(cp_nom.size()) != expected) return;
    HANDLE_ERROR(cudaMemcpy(d_cp_nom_, cp_nom.data(),
        expected * sizeof(float), cudaMemcpyHostToDevice));
  }

  // --- Core MPPI loop ---

  /**
   * @brief Run the B-Spline MPPI optimization.
   *
   * Steps:
   * 1. Copy state to device
   * 2. Expand nominal CPs → u_nom (for visualization / get_action)
   * 3. Sample N(0,1) noise in CP space [K × n_cp × nu]
   * 4. Optionally bias α fraction toward reference CPs
   * 5. Build perturbed CPs: cp_nom + noise * sigma_cp
   * 6. Batch-expand perturbed CPs → u_expanded [K × H × nu]
   * 7. Rollout all K trajectories with pre-computed controls
   * 8. Compute softmax importance weights
   * 9. Update cp_nom via weighted CP-space noise
   *
   * @param state  Current state vector.
   */
  void compute(const Eigen::VectorXf& state)
  {
    HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(),
        config_.nx * sizeof(float), cudaMemcpyHostToDevice));

    const int cp_total = bspline_.n_cp * config_.nu;

    // 1. Expand nominal CPs → u_nom
    bspline_.expand(d_cp_nom_, d_u_nom_, 1, config_.horizon,
                    config_.nu, config_.dt);

    // 2. Sample N(0,1) noise in CP space
    // curandGenerateNormal requires even count
    int noise_count = config_.num_samples * cp_total;
    int noise_alloc = noise_count + (noise_count % 2);
    HANDLE_CURAND_ERROR(curandGenerateNormal(gen_, d_cp_noise_,
        noise_alloc, 0.0f, 1.0f));

    // 3. Biased sampling: shift alpha fraction toward reference CPs
    if (has_cp_ref_ && config_.alpha > 0.0f) {
      int num_biased = static_cast<int>(config_.num_samples * config_.alpha);
      int start_biased = config_.num_samples - num_biased;
      if (num_biased > 0) {
        dim3 block(256);
        dim3 grid((config_.num_samples + block.x - 1) / block.x);
        apply_cp_bias_kernel<<<grid, block>>>(
            d_cp_noise_, d_cp_nom_, d_cp_ref_, d_sigma_cp_,
            config_.num_samples, bspline_.n_cp, config_.nu,
            start_biased);
        HANDLE_ERROR(cudaGetLastError());
      }
    }

    // 4. Build perturbed CPs: cp_nom + noise * sigma_cp
    {
      int total = config_.num_samples * cp_total;
      dim3 block(256);
      dim3 grid((total + block.x - 1) / block.x);
      build_cp_samples_kernel<<<grid, block>>>(
          d_cp_nom_, d_cp_noise_, d_cp_samples_, d_sigma_cp_,
          config_.num_samples, bspline_.n_cp, config_.nu);
      HANDLE_ERROR(cudaGetLastError());
    }

    // 5. Batch-expand perturbed CPs → u_expanded [K × H × nu]
    bspline_.expand(d_cp_samples_, d_u_expanded_,
                    config_.num_samples, config_.horizon,
                    config_.nu, config_.dt);

    // 6. Rollout with pre-computed controls
    {
      dim3 block(256);
      dim3 grid((config_.num_samples + block.x - 1) / block.x);
      rollout_bspline_kernel<<<grid, block>>>(
          dynamics_, cost_, config_,
          d_initial_state_, d_u_expanded_, d_u_applied_,
          d_costs_);
      HANDLE_ERROR(cudaGetLastError());
      HANDLE_ERROR(cudaDeviceSynchronize());
    }

    // 7. Compute softmax weights (host)
    std::vector<float> h_costs(config_.num_samples);
    HANDLE_ERROR(cudaMemcpy(h_costs.data(), d_costs_,
        config_.num_samples * sizeof(float), cudaMemcpyDeviceToHost));

    float min_cost = *std::min_element(h_costs.begin(), h_costs.end());
    std::vector<float> h_weights(config_.num_samples);
    float sum_weights = 0.0f;
    for (int k = 0; k < config_.num_samples; ++k) {
      float w = expf(-(h_costs[k] - min_cost) / config_.lambda);
      h_weights[k] = w;
      sum_weights += w;
    }
    for (float& w : h_weights) w /= sum_weights;

    HANDLE_ERROR(cudaMemcpy(d_weights_, h_weights.data(),
        config_.num_samples * sizeof(float), cudaMemcpyHostToDevice));

    // 8. Update control points via weighted CP noise
    {
      int threads = 256;
      int blocks = (cp_total + threads - 1) / threads;
      weighted_update_cp_kernel<<<blocks, threads>>>(
          d_cp_nom_, d_cp_noise_, d_weights_, d_sigma_cp_,
          config_.num_samples, cp_total, config_.nu,
          config_.learning_rate);
      HANDLE_ERROR(cudaGetLastError());
      HANDLE_ERROR(cudaDeviceSynchronize());
    }

    // Re-expand nominal for get_action / get_planned_trajectory
    bspline_.expand(d_cp_nom_, d_u_nom_, 1, config_.horizon,
                    config_.nu, config_.dt);
  }

  /**
   * @brief Retrieve the first optimal control action.
   *
   * @return First control from the expanded nominal sequence.
   */
  Eigen::VectorXf get_action()
  {
    Eigen::VectorXf action(config_.nu);
    HANDLE_ERROR(cudaMemcpy(action.data(), d_u_nom_,
        config_.nu * sizeof(float), cudaMemcpyDeviceToHost));
    return action;
  }

  /**
   * @brief Shift control points forward by one position.
   *
   * Removes the first CP and holds the last one constant.
   */
  void shift()
  {
    int total = bspline_.n_cp * config_.nu;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    shift_cp_kernel<<<blocks, threads>>>(d_cp_nom_, bspline_.n_cp, config_.nu);
    HANDLE_ERROR(cudaGetLastError());
  }

  const MPPIConfig& config() const { return config_; }
  const BSplineConfig& bspline_config() const { return bspline_; }

private:
  MPPIConfig config_;
  Dynamics dynamics_;
  Cost cost_;
  BSplineConfig bspline_;

  // Device buffers
  float* d_cp_nom_;         ///< Nominal CPs [n_cp × nu].
  float* d_cp_noise_;       ///< CP noise [K × n_cp × nu].
  float* d_cp_samples_;     ///< Perturbed CPs [K × n_cp × nu].
  float* d_u_expanded_;     ///< Expanded controls [K × H × nu].
  float* d_u_nom_;          ///< Expanded nominal [H × nu].
  float* d_cp_ref_;         ///< Reference CPs [n_cp × nu] (for biased sampling).
  float* d_sigma_cp_;       ///< CP noise sigma [nu].
  float* d_costs_;          ///< Per-sample costs [K].
  float* d_weights_;        ///< Softmax weights [K].
  float* d_initial_state_;  ///< Current state [nx].
  float* d_u_applied_;      ///< Last applied control [nu].
  bool has_cp_ref_ = false;

  curandGenerator_t gen_;
};

}  // namespace mppi

#endif  // BSPLINE_MPPI_CONTROLLER_CUH
