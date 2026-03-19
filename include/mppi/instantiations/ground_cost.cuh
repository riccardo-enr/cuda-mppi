/**
 * @file ground_cost.cuh
 * @brief Multi-layer cost function for UGV informative MPPI.
 *
 * For 4D bicycle state $[x, y, \theta, v]$ with 2D control
 * $[\text{throttle}, \text{steering}]$ (both normalised $[-1, 1]$).
 *
 * | # | Layer                      | Weight             | Sign |
 * |---|----------------------------|--------------------|------|
 * | 1 | Grid obstacle collision    | `collision_penalty`| +    |
 * | 2 | Workspace bounds           | `collision_penalty`| +    |
 * | 3 | Goal attraction (nearest)  | `goal_weight`      | +    |
 * | 4 | Uniform-FSMI local info    | `lambda_local`     | -    |
 * | 5 | Info field lookup          | `lambda_info`      | -    |
 * | 6 | Action regularisation      | `action_reg`       | +    |
 * | 7 | Velocity penalty           | `vel_weight`       | +    |
 * | 8 | Steering penalty           | `steering_weight`  | +    |
 */

#ifndef MPPI_GROUND_COST_CUH
#define MPPI_GROUND_COST_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "mppi/core/map.cuh"
#include "mppi/core/fsmi.cuh"

namespace mppi {
namespace instantiations {

struct GroundCost
{
  /// Map and information structures
  OccupancyGrid2D grid;
  InfoField info_field;
  UniformFSMIConfig uniform_cfg;

  /// Cost weights
  float lambda_info = 5.0f;
  float lambda_local = 10.0f;
  float goal_weight = 0.5f;
  float collision_penalty = 1000.0f;
  float occ_threshold = 0.7f;
  float action_reg = 0.01f;
  float vel_weight = 0.5f;
  float v_ref = 1.0f;          ///< Preferred forward speed (m/s).
  float steering_weight = 0.1f;

  /// Multi-goal support
  static constexpr int MAX_GOALS = 32;
  float2 goals[MAX_GOALS] = {};
  int num_goals = 1;

  /// Workspace bounds (world coordinates)
  float bound_x_min = -5.0f;
  float bound_x_max = 15.0f;
  float bound_y_min = -5.0f;
  float bound_y_max = 15.0f;

  /**
   * @brief Compute the running cost at timestep t.
   *
   * State: [x, y, theta, v]
   * Control: [throttle, steering] (normalised [-1, 1])
   */
  __device__ float compute(const float* x, const float* u,
                           const float* /*u_prev*/, int /*t*/) const
  {
    float cost = 0.0f;
    float px = x[0], py = x[1];
    float theta = x[2], v = x[3];

    // 1. Grid-based obstacle cost
    int2 gi = grid.world_to_grid(make_float2(px, py));
    int idx = grid.get_index(gi.x, gi.y);
    if (idx >= 0) {
      float p = grid.data[idx];
      if (p >= occ_threshold) { cost += collision_penalty; }
    }

    // 2. Bounds cost — ramp margin so MPPI plans turns before hitting wall
    constexpr float margin = 1.5f;
    auto ramp = [](float val, float lo, float hi, float m, float pen) -> float {
        float c = 0.0f;
        if (val < lo)            c += pen;
        else if (val < lo + m)   c += pen * (lo + m - val) / m;
        if (val > hi)            c += pen;
        else if (val > hi - m)   c += pen * (val - hi + m) / m;
        return c;
    };
    cost += ramp(px, bound_x_min, bound_x_max, margin, collision_penalty);
    cost += ramp(py, bound_y_min, bound_y_max, margin, collision_penalty);

    // 3. Goal attraction (min distance to nearest goal)
    float min_goal_dist = 1e10f;
    for (int g = 0; g < num_goals; ++g) {
      float gx = px - goals[g].x, gy = py - goals[g].y;
      float d = sqrtf(gx * gx + gy * gy);
      if (d < min_goal_dist) min_goal_dist = d;
    }
    cost += goal_weight * min_goal_dist;

    // 4. Uniform-FSMI local information reward (theta directly from state)
    float info_gain = compute_uniform_fsmi_at_pose(
        grid, make_float2(px, py), theta, uniform_cfg);
    cost -= lambda_local * info_gain;

    // 5. Info field lookup (strategic guidance)
    if (info_field.d_field != nullptr) {
      float field_val = info_field.sample(make_float2(px, py));
      cost -= lambda_info * field_val;
    }

    // 6. Action regularization
    cost += action_reg * (u[0] * u[0] + u[1] * u[1]);

    // 7. Velocity penalty (encourage preferred speed)
    float dv = v - v_ref;
    cost += vel_weight * dv * dv;

    // 8. Steering penalty (smooth turning)
    cost += steering_weight * u[1] * u[1];

    return cost;
  }

  /**
   * @brief Compute the terminal cost.
   *
   * Strong goal attraction + velocity penalty (want to arrive slow).
   */
  __device__ float terminal_cost(const float* x) const
  {
    float px = x[0], py = x[1];
    float v = x[3];

    // Goal attraction (10x running weight)
    float min_goal_dist = 1e10f;
    for (int g = 0; g < num_goals; ++g) {
      float gx = px - goals[g].x, gy = py - goals[g].y;
      float d = sqrtf(gx * gx + gy * gy);
      if (d < min_goal_dist) min_goal_dist = d;
    }
    float goal_cost = 10.0f * min_goal_dist;

    // Terminal velocity penalty (want near-zero at goal)
    float vel_cost = 5.0f * v * v;

    return goal_cost + vel_cost;
  }
};

}  // namespace instantiations
}  // namespace mppi

#endif  // MPPI_GROUND_COST_CUH
