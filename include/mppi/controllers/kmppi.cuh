#ifndef KMPPI_CONTROLLER_CUH
#define KMPPI_CONTROLLER_CUH

#include "mppi.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace mppi
{

__global__ void interpolation_kernel(
  const float * noise_theta,   // (K, M, nu)
  const float * W,             // (T, M)
  float * noise_interp,        // (K, T, nu)
  int K, int T, int M, int nu
);

__global__ void interpolate_single_kernel(
  const float * theta,   // (M, nu)
  const float * W,       // (T, M)
  float * u_nom,         // (T, nu)
  int T, int M, int nu
);

template<typename Dynamics, typename Cost>
class KMPPIController {
public:
  KMPPIController(const MPPIConfig & config, const Dynamics & dynamics, const Cost & cost)
  : config_(config), dynamics_(dynamics), cost_(cost)
  {

    int num_support = config.num_support_pts;

    HANDLE_ERROR(cudaMalloc(&d_theta_, num_support * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_noise_theta_,
        config.num_samples * num_support * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_noise_interp_,
        config.num_samples * config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_u_nom_, config.horizon * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_interp_matrix_, config.horizon * num_support * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&d_costs_, config.num_samples * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_initial_state_, config.nx * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_weights_, config.num_samples * sizeof(float)));

    HANDLE_ERROR(cudaMemset(d_theta_, 0, num_support * config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_nom_, 0, config.horizon * config.nu * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&d_u_applied_, config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_applied_, 0, config.nu * sizeof(float)));

    HANDLE_CURAND_ERROR(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));

        // Compute Interpolation Matrix (W) on host and copy
    compute_interpolation_matrix();
  }

  ~KMPPIController()
  {
    cudaFree(d_theta_);
    cudaFree(d_noise_theta_);
    cudaFree(d_noise_interp_);
    cudaFree(d_u_nom_);
    cudaFree(d_interp_matrix_);
    cudaFree(d_costs_);
    cudaFree(d_initial_state_);
    cudaFree(d_weights_);
    cudaFree(d_u_applied_);
    curandDestroyGenerator(gen_);
  }

  void compute_interpolation_matrix()
  {
        // Simple RBF kernel implementation on host
    int T = config_.horizon;
    int M = config_.num_support_pts;
    // RBF sigma should scale with support point spacing to ensure smooth
    // interpolation. With spacing = (T-1)/(M-1), sigma ≈ spacing gives
    // good overlap between adjacent basis functions.
    float spacing = (M > 1) ? static_cast<float>(T - 1) / (M - 1) : 1.0f;
    float sigma = spacing;

    Eigen::MatrixXf K(T, M);
    Eigen::MatrixXf Kmm(M, M);

        // Time points
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

        // Weights = K * inv(Kmm) ? No, solve Kmm * W' = K' -> W = K * inv(Kmm)
        // JAX: weights = solve(Ktktk, K.T).T
        // K.T is (M, T). solve gives (M, T). .T gives (T, M).
        // So W = (inv(Kmm) * K')' = K * inv(Kmm)' = K * inv(Kmm) (since sym)

    Eigen::MatrixXf W = K * Kmm.ldlt().solve(Eigen::MatrixXf::Identity(M, M));

        // Copy W to device (row major)
        // Eigen is col major by default, need to be careful or use RowMajor.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> W_row = W;
    HANDLE_ERROR(cudaMemcpy(d_interp_matrix_, W_row.data(), T * M * sizeof(float),
        cudaMemcpyHostToDevice));
  }

  void compute(const Eigen::VectorXf & state)
  {
    HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(), config_.nx * sizeof(float),
        cudaMemcpyHostToDevice));

        // 1. Sample theta noise
    HANDLE_CURAND_ERROR(curandGenerateNormal(gen_, d_noise_theta_,
        config_.num_samples * config_.num_support_pts * config_.nu, 0.0f, 1.0f));

        // 2. Interpolate noise: noise_interp = noise_theta * W^T ?
        // d_interp_matrix_ is (T, M). noise_theta is (K, M, nu).
        // Result noise_interp is (K, T, nu).
        // For each k, each nu: vec_T = W * vec_M.
        // Parallelize over K and Nu. Loop over T and M (matrix vec mult).

    dim3 block(256);
    dim3 grid((config_.num_samples + block.x - 1) / block.x);     // Grid covers samples
        // Actually grid should cover K * Nu.
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

        // 3. Rollout using interpolated noise
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

    // Compute softmax weights
    std::vector<float> h_costs(config_.num_samples);
    HANDLE_ERROR(cudaMemcpy(h_costs.data(), d_costs_, config_.num_samples * sizeof(float),
        cudaMemcpyDeviceToHost));

    const float min_cost = *std::min_element(h_costs.begin(), h_costs.end());

    std::vector<float> h_weights(config_.num_samples);
    float sum_weights = 0.0f;
    for(int k = 0; k < config_.num_samples; ++k) {
      const float w = expf(-(h_costs[k] - min_cost) / config_.lambda);
      h_weights[k] = w;
      sum_weights += w;
    }
    for(float& w : h_weights) {
      w /= sum_weights;
    }

        // 5. Update Theta
    HANDLE_ERROR(cudaMemcpy(d_weights_, h_weights.data(), config_.num_samples * sizeof(float),
        cudaMemcpyHostToDevice));

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

  void set_applied_control(const Eigen::VectorXf& u) {
    HANDLE_ERROR(cudaMemcpy(d_u_applied_, u.data(),
                            config_.nu * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  Eigen::VectorXf get_action()
  {
    Eigen::VectorXf action(config_.nu);
    HANDLE_ERROR(cudaMemcpy(action.data(), d_u_nom_, config_.nu * sizeof(float),
        cudaMemcpyDeviceToHost));
    return action;
  }

private:
  MPPIConfig config_;
  Dynamics dynamics_;
  Cost cost_;

  float * d_theta_;
  float * d_noise_theta_;   // (K, M, nu)
  float * d_noise_interp_;   // (K, T, nu)
  float * d_u_nom_;   // (T, nu)
  float * d_interp_matrix_;   // (T, M)

  float * d_costs_;
  float * d_initial_state_;
  float * d_weights_;
  float * d_u_applied_;

  curandGenerator_t gen_;
};

// Kernels

__global__ void interpolation_kernel(
  const float * noise_theta,   // (K, M, nu)
  const float * W,             // (T, M)
  float * noise_interp,        // (K, T, nu)
  int K, int T, int M, int nu
)
{
    // Thread per (k, i) -> K * nu
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * nu) {return;}

  int k = idx / nu;
  int i = idx % nu;

    // For this sample k and channel i, compute vector over T
    // noise_interp[k, t, i] = sum_m W[t, m] * noise_theta[k, m, i]

  for(int t = 0; t < T; ++t) {
    float sum = 0.0f;
    for(int m = 0; m < M; ++m) {
      float w_val = W[t * M + m];
            // noise_theta index: k * (M * nu) + m * nu + i
      float n_val = noise_theta[k * (M * nu) + m * nu + i];
      sum += w_val * n_val;
    }
        // noise_interp index: k * (T * nu) + t * nu + i
    noise_interp[k * (T * nu) + t * nu + i] = sum;
  }
}

__global__ void interpolate_single_kernel(
  const float * theta,   // (M, nu)
  const float * W,       // (T, M)
  float * u_nom,         // (T, nu)
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

} // namespace mppi

#endif // KMPPI_CONTROLLER_CUH
