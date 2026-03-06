#ifndef MPPI_COMMON_CUH
#define MPPI_COMMON_CUH

#include <cuda_runtime.h>

namespace mppi
{

struct MPPIConfig
{
  int num_samples;        // K
  int horizon;            // T
  int nx;                 // State dimension
  int nu;                 // Control dimension
  float lambda;           // Temperature
  float dt;               // Time step
  float u_scale;          // Control scale (uniform, legacy)
  float control_sigma[12] = {1,1,1,1, 1,1,1,1, 1,1,1,1}; // Per-dim noise std dev

    // SMPPI specific
  float w_action_seq_cost;

    // KMPPI specific
  int num_support_pts;

    // I-MPPI specific
  float lambda_info;      // Information gain weight
  float alpha;            // Biased sampling mixture weight [0, 1]

    // Update step size (standard MPPI = 1.0)
  float learning_rate = 1.0f;

    // Iterative refinement
  int num_iters = 1;                     // Optimization iterations per compute()
  float std_dev_decay = 1.0f;            // Noise std dev decay per iteration

    // Exploration
  float pure_noise_percentage = 0.0f;    // Fraction of samples with zero-mean noise
};

} // namespace mppi

#endif // MPPI_COMMON_CUH
