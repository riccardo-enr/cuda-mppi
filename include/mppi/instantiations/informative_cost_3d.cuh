/**
 * @file informative_cost_3d.cuh
 * @brief Informative cost function for 3D acceleration-controlled I-MPPI.
 *
 * Designed for a 6D point-mass state (position + velocity in NED) with
 * 3D acceleration control.  Paired with `DoubleIntegrator3D` dynamics.
 *
 * | # | Layer                      | Weight             | Sign |
 * |---|----------------------------|--------------------|------|
 * | 1 | Grid obstacle collision    | `collision_penalty`| +    |
 * | 2 | Workspace bounds           | `collision_penalty`| +    |
 * | 3 | Altitude tracking          | `height_weight`    | +    |
 * | 4 | Reference trajectory track | `target_weight`    | +    |
 * | 5 | Uniform-FSMI local info    | `lambda_local`     | −    |
 * | 6 | Info field lookup          | `lambda_info`      | −    |
 * | 7 | Action regularisation      | `action_reg`       | +    |
 *
 * Layer 5 derives yaw from the velocity direction:
 * $\psi = \operatorname{atan2}(v_y, v_x)$.  At near-zero speed the
 * yaw is effectively random, which gives an orientation-averaged MI
 * estimate — reasonable for a hovering explorer.
 *
 * @see `DoubleIntegrator3D`, `InformativeCost` (13D quadrotor variant).
 */

#ifndef MPPI_INFORMATIVE_COST_3D_CUH
#define MPPI_INFORMATIVE_COST_3D_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "mppi/core/map.cuh"
#include "mppi/core/fsmi.cuh"

namespace mppi {
namespace instantiations {

struct InformativeCost3D
{
  /* ── Map and information structures ──────────────────────────────── */

  OccupancyGrid2D grid;          ///< 2D occupancy grid for obstacle/FSMI queries.
  OccupancyGrid   grid_3d;       ///< 3D voxel grid for collision queries.
  bool use_grid_3d = false;      ///< Use 3D grid for collision instead of 2D.
  InfoField info_field;           ///< Precomputed information potential field.
  UniformFSMIConfig uniform_cfg;  ///< Configuration for local uniform-FSMI.

  /* ── Cost weights ────────────────────────────────────────────────── */

  float lambda_info = 5.0f;      ///< Info field lookup weight (layer 6).
  float lambda_local = 10.0f;    ///< Uniform-FSMI weight (layer 5).
  float target_weight = 1.0f;    ///< Reference trajectory tracking (layer 4).

  /* ── Obstacle / collision ────────────────────────────────────────── */

  float collision_penalty = 1000.0f;
  float occ_threshold = 0.7f;

  /* ── Height tracking ─────────────────────────────────────────────── */

  float height_weight = 10.0f;
  float target_altitude = -2.0f;  ///< NED (negative = up).

  /* ── Action regularisation ───────────────────────────────────────── */

  float action_reg = 0.01f;       ///< $\ell_2$ penalty on acceleration.
  float velocity_weight = 0.0f;
  float max_velocity = 5.0f;

  /* ── Workspace bounds ────────────────────────────────────────────── */

  float bound_x_min = -1.0f;
  float bound_x_max = 14.0f;
  float bound_y_min = -1.0f;
  float bound_y_max = 11.0f;

  /* ── Reference trajectory ────────────────────────────────────────── */

  const float * ref_trajectory = nullptr;  ///< Device pointer, $[H \times 3]$.
  int          ref_horizon = 0;

  /**
   * @brief Running cost for state $[p_x,p_y,p_z,v_x,v_y,v_z]$
   *        and control $[a_x,a_y,a_z]$.
   */
  __device__ float compute(const float * x, const float * u,
                           const float * /*u_prev*/, int t) const
  {
    float cost = 0.0f;
    float px = x[0], py = x[1], pz = x[2];

    /* 1. Grid-based obstacle cost (continuous: scales with probability) */
    if (use_grid_3d) {
      float p = grid_3d.get_probability(make_float3(px, py, pz));
      if (p >= occ_threshold) {
        cost += collision_penalty * (p / occ_threshold);
      }
    } else {
      int2 gi = grid.world_to_grid(make_float2(px, py));
      int idx = grid.get_index(gi.x, gi.y);
      if (idx >= 0) {
        float p = grid.data[idx];
        if (p >= occ_threshold) {
          cost += collision_penalty * (p / occ_threshold);
        }
      }
    }

    /* 2. Bounds cost */
    {
      float dx_lo = fmaxf(bound_x_min - px, 0.0f);
      float dx_hi = fmaxf(px - bound_x_max, 0.0f);
      float dy_lo = fmaxf(bound_y_min - py, 0.0f);
      float dy_hi = fmaxf(py - bound_y_max, 0.0f);
      float d2 = dx_lo * dx_lo + dx_hi * dx_hi +
                 dy_lo * dy_lo + dy_hi * dy_hi;
      if (d2 > 0.0f) { cost += collision_penalty * (1.0f + d2); }
    }

    /* 3. Height cost */
    float dz = pz - target_altitude;
    cost += height_weight * dz * dz;

    /* 4. Reference trajectory tracking */
    if (ref_trajectory != nullptr && ref_horizon > 0) {
      int ti = (t < ref_horizon) ? t : (ref_horizon - 1);
      float rx = ref_trajectory[ti * 3 + 0];
      float ry = ref_trajectory[ti * 3 + 1];
      float rz = ref_trajectory[ti * 3 + 2];
      float ddx = px - rx, ddy = py - ry, ddz = pz - rz;
      cost += target_weight * sqrtf(ddx * ddx + ddy * ddy + ddz * ddz);
    }

    /* 5. Uniform-FSMI local information reward.
     *    Yaw = velocity heading: atan2(vy, vx).                      */
    float vx = x[3], vy = x[4];
    float yaw = atan2f(vy, vx);
    float info_gain = compute_uniform_fsmi_at_pose(
            grid, make_float2(px, py), yaw, uniform_cfg);
    cost -= lambda_local * info_gain;

    /* 6. Info field lookup (strategic guidance) */
    if (info_field.d_field != nullptr) {
      float field_val = info_field.sample(make_float2(px, py));
      cost -= lambda_info * field_val;
    }

    /* 7. Action regularisation — 3 acceleration components */
    for (int i = 0; i < 3; ++i) {
      cost += action_reg * u[i] * u[i];
    }

    /* 8. Velocity penalty (soft speed limit) */
    if (velocity_weight > 0.0f) {
      float vz = x[5];
      float speed2 = vx * vx + vy * vy + vz * vz;
      float v_max2 = max_velocity * max_velocity;
      if (speed2 > v_max2) {
        cost += velocity_weight * (speed2 - v_max2);
      }
    }

    return cost;
  }

  /**
   * @brief Terminal cost — height penalty + reference endpoint.
   */
  __device__ float terminal_cost(const float * x) const
  {
    float cost = 0.0f;

    /* Height */
    float dz = x[2] - target_altitude;
    cost += height_weight * dz * dz;

    /* Reference endpoint (if available) */
    if (ref_trajectory != nullptr && ref_horizon > 0) {
      int ti = ref_horizon - 1;
      float rx = ref_trajectory[ti * 3 + 0];
      float ry = ref_trajectory[ti * 3 + 1];
      float rz = ref_trajectory[ti * 3 + 2];
      float dx = x[0] - rx, dy = x[1] - ry, ddz = x[2] - rz;
      cost += 10.0f * target_weight * sqrtf(dx * dx + dy * dy + ddz * ddz);
    }

    return cost;
  }
};

}  // namespace instantiations
}  // namespace mppi

#endif  // MPPI_INFORMATIVE_COST_3D_CUH
