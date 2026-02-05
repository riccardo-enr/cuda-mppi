#ifndef MPPI_COMMON_CUH
#define MPPI_COMMON_CUH

#include <cuda_runtime.h>

namespace mppi {

struct MPPIConfig {
    int num_samples;      // K
    int horizon;          // T
    int nx;               // State dimension
    int nu;               // Control dimension
    float lambda;         // Temperature
    float dt;             // Time step
    float u_scale;        // Control scale
    
    // SMPPI specific
    float w_action_seq_cost; 

    // KMPPI specific
    int num_support_pts;
    
    // I-MPPI specific
    float lambda_info;    // Information gain weight
    float alpha;          // Biased sampling mixture weight [0, 1]
};

// Simple vector types if needed, or rely on float* and strides
// We'll use raw pointers for now in kernels

} // namespace mppi

#endif // MPPI_COMMON_CUH
