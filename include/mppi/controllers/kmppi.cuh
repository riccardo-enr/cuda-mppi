/**
 * @file kmppi.cuh
 * @brief Kernel-based MPPI (K-MPPI) controller with RBF interpolation.
 *
 * Instead of optimising the full $T \times n_u$ control sequence directly,
 * K-MPPI parameterises the trajectory through $M$ support (knot) points
 * $\boldsymbol{\theta} \in \mathbb{R}^{M \times n_u}$ and reconstructs
 * the dense sequence via radial basis function (RBF) interpolation:
 *
 * $$
 *   \mathbf{u}[t] = \sum_{m=0}^{M-1} W[t, m] \, \boldsymbol{\theta}[m]
 * $$
 *
 * where $W = K_{T \times M} \, K_{M \times M}^{-1}$ is the interpolation
 * matrix built from Gaussian RBF kernels
 * $k(t_i, t_j) = \exp\!\bigl(-\frac{(t_i - t_j)^2}{2\sigma^2}\bigr)$.
 *
 * This reduces the effective search dimensionality from $T \cdot n_u$
 * to $M \cdot n_u$, producing inherently smooth trajectories.
 *
 * @tparam Dynamics  Dynamics model (GPU-callable).
 * @tparam Cost      Cost function (GPU-callable).
 */

#ifndef KMPPI_CONTROLLER_CUH
#define KMPPI_CONTROLLER_CUH

#include "mppi.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace mppi {

/**
 * @brief CUDA kernel to interpolate noise from support points to full horizon.
 *
 * For each sample $k$ and control channel $i$, computes:
 *
 * $$
 *   \epsilon_{\text{interp}}[k, t, i] = \sum_{m=0}^{M-1} W[t, m] \, \epsilon_\theta[k, m, i]
 * $$
 *
 * One thread per $(k, i)$ pair; loops over $T$ and $M$.
 *
 * @param noise_theta   Support-point noise $[K \times M \times n_u]$.
 * @param W             Interpolation matrix $[T \times M]$.
 * @param noise_interp  Output interpolated noise $[K \times T \times n_u]$.
 * @param K             Number of samples.
 * @param T             Horizon length.
 * @param M             Number of support points.
 * @param nu            Control dimension.
 */
__global__ void interpolation_kernel(
  const float * noise_theta,
  const float * W,
  float * noise_interp,
  int K, int T, int M, int nu
);

/**
 * @brief CUDA kernel to interpolate a single set of support points to a full sequence.
 *
 * Computes $\mathbf{u}[t, i] = \sum_m W[t, m] \, \theta[m, i]$.
 * One thread per $(t, i)$.
 *
 * @param theta  Support-point values $[M \times n_u]$.
 * @param W      Interpolation matrix $[T \times M]$.
 * @param u_nom  Output control sequence $[T \times n_u]$.
 * @param T      Horizon length.
 * @param M      Number of support points.
 * @param nu     Control dimension.
 */
__global__ void interpolate_single_kernel(
  const float * theta,
  const float * W,
  float * u_nom,
  int T, int M, int nu
);

/**
 * @brief K-MPPI controller with RBF kernel interpolation.
 *
 * Optimises in the low-dimensional support-point space and interpolates
 * back to the full horizon for rollout evaluation.
 *
 * @tparam Dynamics  Dynamics model (GPU-callable).
 * @tparam Cost      Cost function (GPU-callable).
 */
template<typename Dynamics, typename Cost>
class KMPPIController {
public:
  /**
   * @brief Construct the K-MPPI controller.
   *
   * Allocates device memory for support points, interpolated noise,
   * and the interpolation matrix $W$. Computes $W$ on the host
   * using Gaussian RBF kernels.
   *
   * @param config    MPPI config (uses `num_support_pts` for $M$).
   * @param dynamics  Dynamics model instance.
   * @param cost      Cost function instance.
   */
  KMPPIController(const MPPIConfig & config, const Dynamics & dynamics, const Cost & cost)
  : config_(config), dynamics_(dynamics), cost_(cost)
  {

    int num_support = config.num_support_pts;

    HANDLE_ERROR(cudaMalloc(&d_theta_, num_support * config.nu * sizeof(float)));
    /* curandGenerateNormal requires an even element count. */
    size_t noise_count_ = static_cast<size_t>(config.num_samples) * num_support * config.nu;
    noise_count_ += (noise_count_ & 1u);
    HANDLE_ERROR(cudaMalloc(&d_noise_theta_, noise_count_ * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_noise_interp_,
        config.num_samples * config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_u_nom_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_interp_matrix_, config.horizon * num_support * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&d_costs_, config.num_samples * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_initial_state_, config.nx * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_weights_, config.num_samples * sizeof(float)));

    HANDLE_ERROR(cudaMemset(d_theta_, 0, num_support * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_nom_, 0, config.horizon * config.nu * sizeof(float)));

    // Reference bias buffer (for split sampling)
    HANDLE_ERROR(cudaMalloc(&d_u_ref_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_ref_, 0, config.horizon * config.nu * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&d_u_applied_, config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_applied_, 0, config.nu * sizeof(float)));

    HANDLE_CURAND_ERROR(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));

    compute_interpolation_matrix();
    softmax_.allocate(config.num_samples);
  }

  /** @brief Destructor. Frees all device buffers. */
  ~KMPPIController()
  {
    cudaFree(d_theta_);
    cudaFree(d_noise_theta_);
    cudaFree(d_noise_interp_);
    cudaFree(d_u_nom_);
    cudaFree(d_u_ref_);
    cudaFree(d_interp_matrix_);
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
   * When set, a fraction `alpha` of noise samples will be shifted toward
   * this reference after interpolation, implementing split sampling.
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
  void clear_reference_sequence() {
    has_ref_bias_ = false;
  }

  /**
   * @brief Compute the RBF interpolation matrix $W$ on the host.
   *
   * Places $M$ support points uniformly across $[0, T-1]$ and builds:
   *
   * $$
   *   W = K_{T \times M} \, K_{M \times M}^{-1}
   * $$
   *
   * where $K_{ij} = \exp\!\bigl(-\frac{(t_i - t_j)^2}{2\sigma^2}\bigr)$
   * and $\sigma$ equals the support-point spacing for smooth overlap.
   */
  void compute_interpolation_matrix()
  {
    int T = config_.horizon;
    int M = config_.num_support_pts;
    float spacing = (M > 1) ? static_cast<float>(T - 1) / (M - 1) : 1.0f;
    float sigma = spacing;

    Eigen::MatrixXf K(T, M);
    Eigen::MatrixXf Kmm(M, M);

    Eigen::VectorXf t = Eigen::VectorXf::LinSpaced(T, 0, T - 1);
    Eigen::VectorXf tk = Eigen::VectorXf::LinSpaced(M, 0, T - 1);

    auto rbf = [&](float t1, float t2) {
        return std::exp(-std::pow(t1 - t2, 2) / (2 * sigma * sigma + 1e-8));
      };

    for(int i = 0; i < T; ++i) {
      for(int j = 0; j < M; ++j) {
        K(i, j) = rbf(t(i), tk(j));
      }
    }

    for(int i = 0; i < M; ++i) {
      for(int j = 0; j < M; ++j) {
        Kmm(i, j) = rbf(tk(i), tk(j));
      }
    }

    Eigen::MatrixXf W = K * Kmm.ldlt().solve(Eigen::MatrixXf::Identity(M, M));

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W_row = W;
    HANDLE_ERROR(cudaMemcpy(d_interp_matrix_, W_row.data(), T * M * sizeof(float),
        cudaMemcpyHostToDevice));
  }

  /**
   * @brief Run the K-MPPI optimization loop.
   *
   * 1. Sample noise in support-point space $[K \times M \times n_u]$
   * 2. Interpolate to full horizon $[K \times T \times n_u]$
   * 3. Rollout all $K$ trajectories
   * 4. Compute softmax weights, update $\boldsymbol{\theta}$
   * 5. Interpolate updated $\boldsymbol{\theta}$ to get $\mathbf{u}_{\text{nom}}$
   *
   * @param state  Current state $\mathbf{x} \in \mathbb{R}^{n_x}$.
   */
  void compute(const Eigen::VectorXf & state)
  {
    HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(), config_.nx * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t noise_n = static_cast<size_t>(config_.num_samples) * config_.num_support_pts * config_.nu;
    noise_n += (noise_n & 1u);
    HANDLE_CURAND_ERROR(curandGenerateNormal(gen_, d_noise_theta_, noise_n, 0.0f, 1.0f));

    dim3 block(256);
    dim3 grid((config_.num_samples + block.x - 1) / block.x);
    int total_channels = config_.num_samples * config_.nu;
    dim3 grid_interp((total_channels + 256 - 1) / 256);

    interpolation_kernel << < grid_interp, 256 >> > (
      d_noise_theta_,
      d_interp_matrix_,
      d_noise_interp_,
      config_.num_samples,
      config_.horizon,
      config_.num_support_pts,
      config_.nu
      );
    HANDLE_ERROR(cudaGetLastError());

    // Apply reference bias to alpha fraction of samples (split sampling)
    if (has_ref_bias_ && config_.alpha > 0.0f) {
      int num_biased = static_cast<int>(config_.num_samples * config_.alpha);
      int start_biased_idx = config_.num_samples - num_biased;
      if (num_biased > 0) {
        dim3 bias_block(256);
        dim3 bias_grid((config_.num_samples + bias_block.x - 1) / bias_block.x);
        kernels::apply_bias_kernel<<<bias_grid, bias_block>>>(
            d_noise_interp_, d_u_nom_, d_u_ref_, config_.num_samples,
            config_.horizon, config_.nu, start_biased_idx);
        HANDLE_ERROR(cudaGetLastError());
      }
    }

    kernels::rollout_kernel << < grid, block >> > (
      dynamics_,
      cost_,
      config_,
      d_initial_state_,
      d_u_nom_,
      d_noise_interp_,
      d_u_applied_,
      d_costs_
      );
    HANDLE_ERROR(cudaGetLastError());

    /* Compute softmax weights on device (no PCIe round-trip). */
    softmax_.compute(d_costs_, d_weights_, config_.lambda, config_.num_samples);

    int num_params = config_.num_support_pts * config_.nu;
    int blocks_upd = (num_params + 256 - 1) / 256;

    weighted_update_kernel << < blocks_upd, 256 >> > (
      d_theta_,
      d_noise_theta_,
      d_weights_,
      config_.num_samples,
      num_params,
      config_
      );

    // Interpolate updated theta to get u_nom = W * theta
    const int interp_threads = (config_.horizon * config_.nu + 255) / 256;
    interpolate_single_kernel<<<interp_threads, 256>>>(
      d_theta_,
      d_interp_matrix_,
      d_u_nom_,
      config_.horizon,
      config_.num_support_pts,
      config_.nu
      );
  }

  /** @brief Upload the last applied control (for rate cost). */
  void set_applied_control(const Eigen::VectorXf& u) {
    HANDLE_ERROR(cudaMemcpy(d_u_applied_, u.data(),
                            config_.nu * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  /**
   * @brief Retrieve the first optimal control action.
   * @return First control $\mathbf{u}[0] \in \mathbb{R}^{n_u}$.
   */
  Eigen::VectorXf get_action()
  {
    Eigen::VectorXf action(config_.nu);
    HANDLE_ERROR(cudaMemcpy(action.data(), d_u_nom_, config_.nu * sizeof(float),
        cudaMemcpyDeviceToHost));
    return action;
  }

private:
  MPPIConfig config_;     ///< MPPI hyperparameters.
  Dynamics dynamics_;     ///< Dynamics model.
  Cost cost_;             ///< Cost function.

  float * d_theta_;           ///< Support-point parameters $[M \times n_u]$.
  float * d_noise_theta_;     ///< Support-point noise $[K \times M \times n_u]$.
  float * d_noise_interp_;    ///< Interpolated noise $[K \times T \times n_u]$.
  float * d_u_nom_;           ///< Dense nominal sequence $[T \times n_u]$.
  float * d_interp_matrix_;   ///< RBF interpolation matrix $W$ $[T \times M]$.

  float * d_u_ref_;           ///< Reference bias sequence $[T \times n_u]$.

  float * d_costs_;           ///< Per-sample rollout costs $[K]$.
  float * d_initial_state_;   ///< Current state $[n_x]$.
  float * d_weights_;         ///< Softmax importance weights $[K]$.
  float * d_u_applied_;       ///< Last applied control $[n_u]$.
  bool has_ref_bias_ = false; ///< Whether reference bias is active.
  SoftmaxWeights softmax_;    ///< GPU-side softmax helper (CUB reductions).

  curandGenerator_t gen_;     ///< CuRAND generator.
};

// ===========================================================================
// Kernel Implementations
// ===========================================================================

__global__ void interpolation_kernel(
  const float * noise_theta,
  const float * W,
  float * noise_interp,
  int K, int T, int M, int nu
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * nu) {return;}

  int k = idx / nu;
  int i = idx % nu;

  for(int t = 0; t < T; ++t) {
    float sum = 0.0f;
    for(int m = 0; m < M; ++m) {
      float w_val = W[t * M + m];
      float n_val = noise_theta[k * (M * nu) + m * nu + i];
      sum += w_val * n_val;
    }
    noise_interp[k * (T * nu) + t * nu + i] = sum;
  }
}

__global__ void interpolate_single_kernel(
  const float * theta,
  const float * W,
  float * u_nom,
  int T, int M, int nu
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * nu) { return; }

  int t = idx / nu;
  int i = idx % nu;

  float sum = 0.0f;
  for (int m = 0; m < M; ++m) {
    sum += W[t * M + m] * theta[m * nu + i];
  }
  u_nom[idx] = sum;
}

}  // namespace mppi

#endif  // KMPPI_CONTROLLER_CUH
