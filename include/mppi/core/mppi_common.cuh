/**
 * @file mppi_common.cuh
 * @brief Shared configuration structure for all MPPI controller variants.
 *
 * Defines `MPPIConfig`, the central hyperparameter struct passed (by value)
 * to GPU kernels. All MPPI variants (standard, SMPPI, KMPPI, I-MPPI) share
 * this struct and ignore the fields they do not use.
 */

#ifndef MPPI_COMMON_CUH
#define MPPI_COMMON_CUH

#include <cuda_runtime.h>

namespace mppi {

/**
 * @brief Hyperparameters for MPPI and its variants.
 *
 * Passed by value to CUDA kernels so that all fields reside in fast GPU
 * registers. The struct is intentionally kept POD-like with fixed-size
 * arrays to guarantee device compatibility.
 *
 * ## Core parameters
 *
 * | Symbol     | Field              | Description                            |
 * |------------|--------------------|----------------------------------------|
 * | $K$        | `num_samples`      | Number of sampled rollouts             |
 * | $T$        | `horizon`          | Prediction horizon length              |
 * | $n_x$      | `nx`               | State dimensionality                   |
 * | $n_u$      | `nu`               | Control dimensionality ($\leq 12$)     |
 * | $\lambda$  | `lambda`           | Temperature / inverse sensitivity      |
 * | $\Delta t$ | `dt`               | Integration time step                  |
 * | $\sigma_i$ | `control_sigma[i]` | Per-dimension noise standard deviation |
 * |            | `u_scale`          | Uniform control scale (legacy)         |
 *
 * ## Variant-specific parameters
 *
 * | Symbol       | Field                | Description                                  |
 * |--------------|----------------------|----------------------------------------------|
 * |              | `w_action_seq_cost`  | SMPPI: action-sequence smoothness weight      |
 * |              | `num_support_pts`    | KMPPI: number of support (knot) points        |
 * | $\lambda_I$  | `lambda_info`        | I-MPPI: information gain weight               |
 * | $\alpha$     | `alpha`              | I-MPPI: biased sampling mixture weight [0, 1] |
 *
 * ## Optimization tuning
 *
 * | Symbol | Field                    | Description                                        |
 * |--------|--------------------------|----------------------------------------------------|
 * |        | `learning_rate`          | Update step size (1.0 = full update)               |
 * |        | `num_iters`              | Refinement iterations per `compute()` call         |
 * |        | `std_dev_decay`          | Multiplicative $\sigma$ decay per iteration        |
 * |        | `pure_noise_percentage`  | Fraction of samples using zero-mean exploration    |
 */
struct MPPIConfig {
  /// @name Core MPPI parameters
  /// @{
  int num_samples;  ///< $K$ — number of sampled rollouts.
  int horizon;      ///< $T$ — prediction horizon length.
  int nx;           ///< $n_x$ — state dimension.
  int nu;           ///< $n_u$ — control dimension ($\leq 12$).
  float lambda;     ///< $\lambda$ — temperature (inverse sensitivity).
  float dt;         ///< $\Delta t$ — integration time step.
  float u_scale;    ///< Uniform control scale (legacy, prefer `control_sigma`).
  float control_sigma[12] = {
      1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1};  ///< $\sigma_i$ — per-dimension noise std dev.
  /// @}

  /// @name Variant-specific parameters
  /// @{
  float
      w_action_seq_cost;  ///< SMPPI: weight on action-sequence smoothness cost.
  int num_support_pts;    ///< KMPPI: number of support (knot) points.
  float lambda_info;      ///< I-MPPI: information gain weight $\lambda_I$.
  float alpha;  ///< I-MPPI: biased sampling mixture weight $\alpha \in [0, 1]$.
  /// @}

  /// @name Optimization tuning
  /// @{
  float learning_rate = 1.0f;  ///< Update step size (standard MPPI = 1.0).
  int num_iters = 1;           ///< Refinement iterations per `compute()` call.
  float std_dev_decay = 1.0f;  ///< Multiplicative $\sigma$ decay per iteration.
  float pure_noise_percentage =
      0.0f;  ///< Fraction of samples using zero-mean (exploration) noise.
  /// @}
};

}  // namespace mppi

#endif  // MPPI_COMMON_CUH
