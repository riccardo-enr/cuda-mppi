/**
 * @file softmax.cuh
 * @brief GPU-side MPPI softmax weight computation via CUB.
 *
 * Replaces the host-side softmax pattern (D2H copy, CPU reduction, H2D copy)
 * with a fully on-device pipeline:
 *
 * $$
 *   w_k = \frac{\exp\!\left(-(S_k - S_{\min}) / \lambda\right)}
 *              {\sum_j \exp\!\left(-(S_j - S_{\min}) / \lambda\right)}
 * $$
 *
 * The four GPU operations (CUB min-reduce, exp kernel, CUB sum-reduce,
 * normalize kernel) run sequentially on the caller's stream so the result
 * is ready for `weighted_update_kernel` without any host round-trips.
 *
 * CUB is header-only and ships with the CUDA toolkit — no extra CMake
 * linkage is required.
 */

#ifndef MPPI_SOFTMAX_CUH
#define MPPI_SOFTMAX_CUH

#include <cuda_runtime.h>

#include <cub/device/device_reduce.cuh>

#include "mppi/utils/cuda_utils.cuh"

namespace mppi {

/* Compute exp-weights: weights[k] = exp(-(costs[k] - *d_min) / lambda). */
__global__ inline void exp_weights_kernel(const float* __restrict__ costs,
                                          float* __restrict__ weights,
                                          const float* __restrict__ d_min,
                                          float lambda, int K) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) {
    return;
  }
  weights[k] = expf(-(costs[k] - *d_min) / lambda);
}

/* Normalize: weights[k] /= *d_sum. */
__global__ inline void normalize_weights_kernel(float* __restrict__ weights,
                                                const float* __restrict__ d_sum,
                                                int K) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) {
    return;
  }
  weights[k] /= *d_sum;
}

/**
 * @brief Manages device memory for GPU-side MPPI softmax computation.
 *
 * Allocate once in the controller constructor, then call `compute()` each
 * MPPI iteration. All operations stay on the GPU — no PCIe transfers.
 *
 * ### Usage
 * ```cpp
 * SoftmaxWeights sm;
 * sm.allocate(config.num_samples);
 * // ... in compute() loop:
 * sm.compute(d_costs_, d_weights_, iter_config.lambda, config_.num_samples);
 * // destructor or explicit:
 * sm.free();
 * ```
 */
struct SoftmaxWeights {
  float* d_min = nullptr;  ///< Device scalar: minimum cost over K samples.
  float* d_sum = nullptr;  ///< Device scalar: sum of exp-weights.
  void* d_temp = nullptr;  ///< CUB temporary reduction storage.
  size_t temp_bytes = 0;   ///< Size of the CUB temp buffer.
  int max_K = 0;           ///< K this was allocated for.

  /**
   * @brief Allocate all device buffers for a given number of samples.
   *
   * Queries CUB for the required temp-storage size for both min and sum
   * reductions, then allocates a single shared buffer sized to the larger.
   *
   * @param K  Number of MPPI samples (must be > 0).
   */
  void allocate(int K) {
    max_K = K;
    HANDLE_ERROR(cudaMalloc(&d_min, sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_sum, sizeof(float)));

    /* Query CUB temp sizes (pass nullptr to get size only). */
    size_t bytes_min = 0, bytes_sum = 0;
    cub::DeviceReduce::Min(nullptr, bytes_min, static_cast<float*>(nullptr),
                           d_min, K);
    cub::DeviceReduce::Sum(nullptr, bytes_sum, static_cast<float*>(nullptr),
                           d_sum, K);
    temp_bytes = (bytes_min > bytes_sum) ? bytes_min : bytes_sum;
    HANDLE_ERROR(cudaMalloc(&d_temp, temp_bytes));
  }

  /**
   * @brief Free all device buffers.
   *
   * Safe to call even if `allocate()` was never called (all pointers
   * are null-initialized).
   */
  void free() {
    if (d_min) {
      cudaFree(d_min);
      d_min = nullptr;
    }
    if (d_sum) {
      cudaFree(d_sum);
      d_sum = nullptr;
    }
    if (d_temp) {
      cudaFree(d_temp);
      d_temp = nullptr;
    }
    temp_bytes = 0;
    max_K = 0;
  }

  /**
   * @brief Compute softmax weights entirely on the GPU.
   *
   * Pipeline:
   * 1. CUB `DeviceReduce::Min` — find $S_{\min}$ over `d_costs[K]`
   * 2. `exp_weights_kernel` — $w_k = \exp(-(S_k - S_{\min}) / \lambda)$
   * 3. CUB `DeviceReduce::Sum` — compute $\sum_k w_k$
   * 4. `normalize_weights_kernel` — $w_k \mathrel{/}= \sum w$
   *
   * All four operations execute on `stream` in order; no host
   * synchronization is needed between them.
   *
   * @param d_costs    Device pointer to per-sample costs $[K]$.
   * @param d_weights  Device pointer to output weights $[K]$ (written).
   * @param lambda     MPPI temperature parameter.
   * @param K          Number of samples (must be $\leq$ `max_K`).
   * @param stream     CUDA stream (default: 0).
   */
  void compute(const float* d_costs, float* d_weights, float lambda, int K,
               cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (K + threads - 1) / threads;

    /* CUB takes size_t& even for the actual call, so use a local copy. */
    size_t bytes = temp_bytes;

    /* 1. Min reduction. */
    cub::DeviceReduce::Min(d_temp, bytes, d_costs, d_min, K, stream);

    /* 2. Exponentiate. */
    exp_weights_kernel<<<blocks, threads, 0, stream>>>(d_costs, d_weights,
                                                       d_min, lambda, K);

    /* 3. Sum reduction. */
    cub::DeviceReduce::Sum(d_temp, bytes, d_weights, d_sum, K, stream);

    /* 4. Normalize. */
    normalize_weights_kernel<<<blocks, threads, 0, stream>>>(d_weights, d_sum,
                                                             K);
  }
};

}  // namespace mppi

#endif  // MPPI_SOFTMAX_CUH
