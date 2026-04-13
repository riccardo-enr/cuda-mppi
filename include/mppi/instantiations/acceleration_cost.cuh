/**
 * @file acceleration_cost.cuh
 * @brief Quadratic tracking cost for PointMass3D (acceleration-command dynamics).
 *
 * Penalises position error, velocity error, control magnitude (regularisation
 * around hover), and control rate-of-change (smoothing).  Optionally checks a
 * 3D occupancy grid for collision penalties.
 *
 * Used by mppi_gtest and mppi_log_trajectories.  Previously lived in the
 * external uav_control repo; moved here to remove the cross-repo dependency.
 */

#ifndef MPPI_ACCELERATION_COST_CUH
#define MPPI_ACCELERATION_COST_CUH

#include <cuda_runtime.h>

#include "mppi/core/map.cuh"

namespace mppi::instantiations {

struct AccelerationTrackingCost {
  /* Reference trajectory: device pointer to [ref_horizon × 6] floats.
   * Each row is [px, py, pz, vx, vy, vz].  nullptr → hover at origin. */
  const float* ref_traj   = nullptr;
  int          ref_horizon = 0;

  float Q_pos  = 10.0f;   ///< Position tracking weight.
  float Q_vel  =  1.0f;   ///< Velocity tracking weight.
  float R_acc  =  0.01f;  ///< Control magnitude regularisation.
  float R_du   =  0.5f;   ///< Control rate-of-change (smoothing).

  float terminal_multiplier = 10.0f;

  /* Optional 3D occupancy grid for collision avoidance. */
  OccupancyGrid grid            = {nullptr, {0, 0, 0}, 0.0f, {0.0f, 0.0f, 0.0f}};
  float         collision_penalty = 100.0f;
  float         occ_threshold     =   0.7f;

  __device__ float compute(const float* x, const float* u,
                           const float* u_prev, int t) const {
    float ref[6] = {};
    if (ref_traj && ref_horizon > 0) {
      int ti = (t < ref_horizon) ? t : (ref_horizon - 1);
      for (int i = 0; i < 6; ++i) ref[i] = ref_traj[ti * 6 + i];
    }

    float dx = x[0]-ref[0], dy = x[1]-ref[1], dz = x[2]-ref[2];
    float cost = Q_pos * (dx*dx + dy*dy + dz*dz);

    float dvx = x[3]-ref[3], dvy = x[4]-ref[4], dvz = x[5]-ref[5];
    cost += Q_vel * (dvx*dvx + dvy*dvy + dvz*dvz);

    cost += R_acc * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

    float du0 = u[0]-u_prev[0], du1 = u[1]-u_prev[1], du2 = u[2]-u_prev[2];
    cost += R_du * (du0*du0 + du1*du1 + du2*du2);

    if (grid.data) {
      float p = grid.get_probability(make_float3(x[0], x[1], x[2]));
      if (p >= occ_threshold) cost += collision_penalty;
    }

    return cost;
  }

  __device__ float terminal_cost(const float* x) const {
    float ref[6] = {};
    if (ref_traj && ref_horizon > 0) {
      int ti = ref_horizon - 1;
      for (int i = 0; i < 6; ++i) ref[i] = ref_traj[ti * 6 + i];
    }

    float dx = x[0]-ref[0], dy = x[1]-ref[1], dz = x[2]-ref[2];
    float dvx = x[3]-ref[3], dvy = x[4]-ref[4], dvz = x[5]-ref[5];
    return terminal_multiplier * (
        Q_pos * (dx*dx + dy*dy + dz*dz) +
        Q_vel * (dvx*dvx + dvy*dvy + dvz*dvz));
  }
};

}  // namespace mppi::instantiations

#endif  // MPPI_ACCELERATION_COST_CUH