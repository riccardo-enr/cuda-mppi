#ifndef MPPI_CONTROLLER_CUH
#define MPPI_CONTROLLER_CUH

#include <cuda_runtime.h>
#include <curand.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "mppi/core/kernels.cuh"
#include "mppi/core/mppi_common.cuh"
#include "mppi/utils/cuda_utils.cuh"

namespace mppi {

__global__ void weighted_update_kernel(float* u_nom, const float* noise,
                                       const float* weights, int K,
                                       int total_params,  // T * nu
                                       MPPIConfig config);

__global__ void weighted_mean_kernel(float* u_nom, const float* noise,
                                     const float* weights, int K,
                                     int total_params,  // T * nu
                                     MPPIConfig config);

__global__ void shift_kernel(float* u_nom, int T, int nu);

template <typename Dynamics, typename Cost>
class MPPIController {
 public:
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

    // Last applied control (for rate-of-change cost)
    HANDLE_ERROR(cudaMalloc(&d_u_applied_, config.nu * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_u_applied_, 0, config.nu * sizeof(float)));

    // Setup CuRAND
    HANDLE_CURAND_ERROR(
        curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));
  }

  ~MPPIController() {
    cudaFree(d_u_nom_);
    cudaFree(d_costs_);
    cudaFree(d_initial_state_);
    cudaFree(d_weights_);
    cudaFree(d_u_applied_);
    curandDestroyGenerator(gen_);
  }

  void set_cost(const Cost& cost) { cost_ = cost; }
  Cost& cost() { return cost_; }
  const Cost& cost() const { return cost_; }
  float* get_u_nom_ptr() { return d_u_nom_; }

  void set_applied_control(const Eigen::VectorXf& u) {
    HANDLE_ERROR(cudaMemcpy(d_u_applied_, u.data(), config_.nu * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  void compute(const Eigen::VectorXf& state) {
    // Copy state to device
    HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(),
                            config_.nx * sizeof(float),
                            cudaMemcpyHostToDevice));

    const int num_iters = config_.num_iters > 0 ? config_.num_iters : 1;
    std::vector<float> h_costs(config_.num_samples);
    std::vector<float> h_weights(config_.num_samples);

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
          gen_, d_noise_,
          config_.num_samples * config_.horizon * config_.nu, 0.0f, 1.0f));

      // Launch Rollout Kernel
      dim3 block(256);
      dim3 grid((config_.num_samples + block.x - 1) / block.x);

      kernels::rollout_kernel<<<grid, block>>>(
          dynamics_, cost_, iter_config, d_initial_state_, d_u_nom_, d_noise_,
          d_u_applied_, d_costs_);
      HANDLE_ERROR(cudaGetLastError());

      // Compute softmax weights on host
      HANDLE_ERROR(cudaMemcpy(h_costs.data(), d_costs_,
                              config_.num_samples * sizeof(float),
                              cudaMemcpyDeviceToHost));

      const float min_cost =
          *std::min_element(h_costs.begin(), h_costs.end());

      float sum_weights = 0.0f;
      for (int k = 0; k < config_.num_samples; ++k) {
        const float w =
            expf(-(h_costs[k] - min_cost) / iter_config.lambda);
        h_weights[k] = w;
        sum_weights += w;
      }
      for (float& w : h_weights) {
        w /= sum_weights;
      }

      HANDLE_ERROR(cudaMemcpy(d_weights_, h_weights.data(),
                              config_.num_samples * sizeof(float),
                              cudaMemcpyHostToDevice));

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
      HANDLE_ERROR(cudaDeviceSynchronize());
    }
  }

  void shift() {
    const int total_floats = config_.horizon * config_.nu;
    const int threads = 256;
    const int blocks = (total_floats + threads - 1) / threads;
    shift_kernel<<<blocks, threads>>>(d_u_nom_, config_.horizon, config_.nu);
  }

  void set_nominal_control(const Eigen::VectorXf& u) {
    // Set each timestep to the same control value
    for (int t = 0; t < config_.horizon; ++t) {
      HANDLE_ERROR(cudaMemcpy(d_u_nom_ + t * config_.nu, u.data(),
                              config_.nu * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
  }

  Eigen::VectorXf get_action() {
    // Return first action from d_u_nom_
    Eigen::VectorXf action(config_.nu);
    HANDLE_ERROR(cudaMemcpy(action.data(), d_u_nom_, config_.nu * sizeof(float),
                            cudaMemcpyDeviceToHost));
    return action;
  }

 protected:
  MPPIConfig config_;
  Dynamics dynamics_;
  Cost cost_;

  // Device memory
  float* d_u_nom_;
  float* d_noise_;
  float* d_costs_;
  float* d_initial_state_;
  float* d_weights_;
  float* d_u_applied_;

  // CuRAND
  curandGenerator_t gen_;
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
                                     int total_params,
                                     MPPIConfig config) {
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
