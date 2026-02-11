#ifndef MPPI_INFORMATIVE_COST_CUH
#define MPPI_INFORMATIVE_COST_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "mppi/core/map.cuh"
#include "mppi/core/fsmi.cuh"

namespace mppi {
namespace instantiations {

// ---------------------------------------------------------------------------
// Helper: extract yaw from quaternion [qw, qx, qy, qz]
// ---------------------------------------------------------------------------
__device__ inline float quat_to_yaw(const float* q) {
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    return atan2f(2.0f * (qw*qz + qx*qy),
                  1.0f - 2.0f * (qy*qy + qz*qz));
}

// ---------------------------------------------------------------------------
// Informative Cost for I-MPPI (Layer 3)
//
// Ported from environment.py::informative_running_cost
//
// Combines:
//   1. Grid-based obstacle cost
//   2. Bounds cost
//   3. Height cost (altitude tracking)
//   4. Reference trajectory tracking
//   5. Uniform-FSMI local information reward
//   6. Info field lookup (strategic guidance)
//   7. Goal attraction
//   8. Action regularization
// ---------------------------------------------------------------------------
struct InformativeCost {
    // Grid for obstacle checking and FSMI computation
    OccupancyGrid2D grid;

    // Precomputed information field (updated at 5 Hz by host)
    InfoField info_field;

    // Uniform-FSMI configuration
    UniformFSMIConfig uniform_cfg;

    // Cost weights
    float lambda_info   = 5.0f;   // info field lookup weight
    float lambda_local  = 10.0f;  // Uniform-FSMI weight
    float target_weight = 1.0f;   // reference trajectory tracking
    float goal_weight   = 0.5f;   // goal attraction

    // Goal position
    float3 goal = {9.0f, 5.0f, -2.0f};

    // Obstacle / collision
    float collision_penalty = 1000.0f;
    float occ_threshold     = 0.7f;

    // Height tracking
    float height_weight    = 10.0f;
    float target_altitude  = -2.0f;  // NED: negative = up

    // Action regularization
    float action_reg = 0.01f;

    // Bounds (world coordinates)
    float bound_x_min = -1.0f;
    float bound_x_max = 14.0f;
    float bound_y_min = -1.0f;
    float bound_y_max = 11.0f;

    // Reference trajectory (device pointer, horizon × 3)
    const float* ref_trajectory = nullptr;
    int          ref_horizon    = 0;

    // -----------------------------------------------------------------------
    __device__ float compute(const float* x, const float* u, int t) const {
        float cost = 0.0f;
        float px = x[0], py = x[1], pz = x[2];

        // 1. Grid-based obstacle cost
        int2 gi = grid.world_to_grid(make_float2(px, py));
        int idx = grid.get_index(gi.x, gi.y);
        if (idx >= 0) {
            float p = grid.data[idx];
            if (p >= occ_threshold) cost += collision_penalty;
        }

        // 2. Bounds cost
        if (px < bound_x_min) cost += collision_penalty;
        if (px > bound_x_max) cost += collision_penalty;
        if (py < bound_y_min) cost += collision_penalty;
        if (py > bound_y_max) cost += collision_penalty;

        // 3. Height cost
        float dz = pz - target_altitude;
        cost += height_weight * dz * dz;

        // 4. Reference trajectory tracking
        if (ref_trajectory != nullptr && ref_horizon > 0) {
            int ti = (t < ref_horizon) ? t : (ref_horizon - 1);
            float rx = ref_trajectory[ti * 3 + 0];
            float ry = ref_trajectory[ti * 3 + 1];
            float rz = ref_trajectory[ti * 3 + 2];
            float ddx = px - rx, ddy = py - ry, ddz = pz - rz;
            cost += target_weight * sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
        }

        // 5. Uniform-FSMI local information reward
        float yaw = quat_to_yaw(x + 6);
        float info_gain = compute_uniform_fsmi_at_pose(
            grid, make_float2(px, py), yaw, uniform_cfg
        );
        cost -= lambda_local * info_gain;

        // 6. Info field lookup (strategic guidance)
        if (info_field.d_field != nullptr) {
            float field_val = info_field.sample(make_float2(px, py));
            cost -= lambda_info * field_val;
        }

        // 7. Goal attraction
        float gx = px - goal.x, gy = py - goal.y, gz = pz - goal.z;
        cost += goal_weight * sqrtf(gx*gx + gy*gy + gz*gz);

        // 8. Action regularization
        for (int i = 0; i < 4; ++i) cost += action_reg * u[i] * u[i];

        return cost;
    }

    // -----------------------------------------------------------------------
    __device__ float terminal_cost(const float* x) const {
        float px = x[0], py = x[1], pz = x[2];

        // Strong goal attraction at terminal
        float gx = px - goal.x, gy = py - goal.y, gz = pz - goal.z;
        float goal_cost = 10.0f * sqrtf(gx*gx + gy*gy + gz*gz);

        // Height penalty
        float dz = pz - target_altitude;
        float h_cost = height_weight * dz * dz;

        return goal_cost + h_cost;
    }
};

} // namespace instantiations
} // namespace mppi

#endif // MPPI_INFORMATIVE_COST_CUH
