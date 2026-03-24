/**
 * @file informative_cost.cuh
 * @brief Multi-layer cost function for Informative MPPI (I-MPPI).
 *
 * Combines eight cost layers evaluated at each timestep to guide
 * exploration while maintaining safety and goal-directed behaviour:
 *
 * | # | Layer                      | Weight             | Sign |
 * |---|----------------------------|--------------------|------|
 * | 1 | Grid obstacle collision    | `collision_penalty`| +    |
 * | 2 | Workspace bounds           | `collision_penalty`| +    |
 * | 3 | Altitude tracking          | `height_weight`    | +    |
 * | 4 | Reference trajectory track | `target_weight`    | +    |
 * | 5 | Uniform-FSMI local info    | `lambda_local`     | −    |
 * | 6 | Info field lookup          | `lambda_info`      | −    |
 * | 7 | Goal attraction (nearest)  | `goal_weight`      | +    |
 * | 8 | Action regularisation      | `action_reg`       | +    |
 */

#ifndef MPPI_INFORMATIVE_COST_CUH
#define MPPI_INFORMATIVE_COST_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "mppi/core/map.cuh"
#include "mppi/core/fsmi.cuh"

namespace mppi {
namespace instantiations {

/**
 * @brief Extract yaw angle from a quaternion $[q_w, q_x, q_y, q_z]$.
 *
 * $$
 *   \psi = \operatorname{atan2}\!\bigl(2(q_w q_z + q_x q_y),\;
 *          1 - 2(q_y^2 + q_z^2)\bigr)
 * $$
 *
 * @param q  Pointer to quaternion array $[q_w, q_x, q_y, q_z]$.
 * @return   Yaw angle in radians.
 */
__device__ inline float quat_to_yaw(const float * q)
{
  float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  return atan2f(2.0f * (qw * qz + qx * qy),
                  1.0f - 2.0f * (qy * qy + qz * qz));
}

/**
 * @brief Informative cost function for I-MPPI exploration.
 *
 * Designed for a 13D quadrotor state
 * $[p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, \omega_x, \omega_y, \omega_z]$
 * with 4D control $[T, \omega_x, \omega_y, \omega_z]$.
 *
 * Supports up to `MAX_GOALS` viewpoints for multi-goal attraction
 * (nearest-viewpoint distance is used).
 */
struct InformativeCost
{
  /// @name Map and information structures
  /// @{
  OccupancyGrid2D grid;         ///< 2D occupancy grid for obstacle/FSMI queries.
  InfoField info_field;          ///< Precomputed information potential field.
  UniformFSMIConfig uniform_cfg; ///< Configuration for local uniform-FSMI.
  /// @}

  /// @name Cost weights
  /// @{
  float lambda_info = 5.0f;     ///< Info field lookup weight (layer 6).
  float lambda_local = 10.0f;   ///< Uniform-FSMI weight (layer 5).
  float target_weight = 1.0f;   ///< Reference trajectory tracking (layer 4).
  float goal_weight = 0.5f;     ///< Goal attraction weight (layer 7).
  /// @}

  /// @name Multi-goal support
  /// @{
  static constexpr int MAX_GOALS = 32;
  float3 goals[MAX_GOALS] = {};  ///< Viewpoint goal positions (NED).
  int num_goals = 1;             ///< Number of active goals.
  /// @}

  /// @name Obstacle / collision
  /// @{
  float collision_penalty = 1000.0f;  ///< Penalty for grid collision or out-of-bounds.
  float occ_threshold = 0.7f;         ///< Occupancy probability threshold.
  /// @}

  /// @name Height tracking
  /// @{
  float height_weight = 10.0f;       ///< Altitude error weight (layer 3).
  float target_altitude = -2.0f;     ///< Target altitude in NED (negative = up).
  /// @}

  /// @name Action regularisation
  /// @{
  float action_reg = 0.01f;          ///< $\ell_2$ penalty on control inputs.
  float velocity_weight = 0.0f;      ///< $\ell_2$ penalty on velocity (layers vx,vy,vz).
  float max_velocity = 5.0f;         ///< Velocity above which penalty applies (m/s).
  /// @}

  /// @name Workspace bounds (world coordinates)
  /// @{
  float bound_x_min = -1.0f;
  float bound_x_max = 14.0f;
  float bound_y_min = -1.0f;
  float bound_y_max = 11.0f;
  /// @}

  /// @name Reference trajectory
  /// @{
  const float * ref_trajectory = nullptr;  ///< Device pointer, $[\text{horizon} \times 3]$.
  int          ref_horizon = 0;            ///< Length of reference trajectory.
  /// @}

  /**
   * @brief Compute the running cost at timestep $t$.
   *
   * Evaluates all 8 cost layers and returns their weighted sum.
   *
   * @param x       Current state $\in \mathbb{R}^{13}$.
   * @param u       Current control $\in \mathbb{R}^{4}$.
   * @param u_prev  Previous control (unused in this cost).
   * @param t       Timestep index.
   * @return        Scalar running cost.
   */
  __device__ float compute(const float * x, const float * u,
                           const float * /*u_prev*/, int t) const
  {
    float cost = 0.0f;
    float px = x[0], py = x[1], pz = x[2];

        // 1. Grid-based obstacle cost
    int2 gi = grid.world_to_grid(make_float2(px, py));
    int idx = grid.get_index(gi.x, gi.y);
    if (idx >= 0) {
      float p = grid.data[idx];
      if (p >= occ_threshold) {cost += collision_penalty;}
    }

        // 2. Bounds cost (soft barrier: quadratic outside, flat penalty at boundary)
    {
      float dx_lo = fmaxf(bound_x_min - px, 0.0f);
      float dx_hi = fmaxf(px - bound_x_max, 0.0f);
      float dy_lo = fmaxf(bound_y_min - py, 0.0f);
      float dy_hi = fmaxf(py - bound_y_max, 0.0f);
      float d2 = dx_lo * dx_lo + dx_hi * dx_hi + dy_lo * dy_lo + dy_hi * dy_hi;
      if (d2 > 0.0f) {cost += collision_penalty * (1.0f + d2);}
    }

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
      cost += target_weight * sqrtf(ddx * ddx + ddy * ddy + ddz * ddz);
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

        // 7. Goal attraction (min distance to nearest viewpoint)
    float min_goal_dist = 1e10f;
    for (int g = 0; g < num_goals; ++g) {
      float gx = px - goals[g].x, gy = py - goals[g].y, gz = pz - goals[g].z;
      float d = sqrtf(gx * gx + gy * gy + gz * gz);
      if (d < min_goal_dist) min_goal_dist = d;
    }
    cost += goal_weight * min_goal_dist;

        // 8. Action regularization
    for (int i = 0; i < 4; ++i) {
      cost += action_reg * u[i] * u[i];
    }

        // 9. Velocity penalty (soft speed limit)
    if (velocity_weight > 0.0f) {
      float vx = x[3], vy = x[4], vz = x[5];
      float speed2 = vx * vx + vy * vy + vz * vz;
      float v_max2 = max_velocity * max_velocity;
      if (speed2 > v_max2) {
        cost += velocity_weight * (speed2 - v_max2);
      }
    }

    return cost;
  }

  /**
   * @brief Compute the terminal cost.
   *
   * Strong goal attraction (nearest viewpoint) plus height penalty.
   *
   * @param x  Terminal state $\in \mathbb{R}^{13}$.
   * @return   Scalar terminal cost.
   */
  __device__ float terminal_cost(const float * x) const
  {
    float px = x[0], py = x[1], pz = x[2];

    float min_goal_dist = 1e10f;
    for (int g = 0; g < num_goals; ++g) {
      float gx = px - goals[g].x, gy = py - goals[g].y, gz = pz - goals[g].z;
      float d = sqrtf(gx * gx + gy * gy + gz * gz);
      if (d < min_goal_dist) min_goal_dist = d;
    }
    float goal_cost = 10.0f * min_goal_dist;

    float dz = pz - target_altitude;
    float h_cost = height_weight * dz * dz;

    return goal_cost + h_cost;
  }
};

}   // namespace instantiations
}  // namespace mppi

#endif  // MPPI_INFORMATIVE_COST_CUH
