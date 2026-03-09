/**
 * @file fsmi_cost.cuh
 * @brief Simple 2D FSMI-based cost function for point-mass / double-integrator systems.
 *
 * Combines goal attraction, control regularisation, velocity penalty,
 * and an omnidirectional (8-beam) entropy-based information reward
 * evaluated on a 3D occupancy grid.
 *
 * This is a lightweight alternative to `InformativeCost` for systems
 * with lower-dimensional state (e.g. 4D double integrator).
 */

#ifndef FSMI_COST_CUH
#define FSMI_COST_CUH

#include <cuda_runtime.h>
#include "mppi/core/mppi_common.cuh"
#include "mppi/core/map.cuh"

namespace mppi {

/**
 * @brief FSMI cost function for 2D information-gathering with a simple dynamics model.
 *
 * Assumes state layout $[p_x, p_y, v_x, v_y, \ldots]$ and 2D control.
 *
 * ## Cost layers
 *
 * 1. **Control regularisation:** $0.1 \, \|\mathbf{u}\|^2$
 * 2. **Goal attraction:** $\lambda_g \, \|\mathbf{p} - \mathbf{g}\|^2$
 * 3. **Velocity penalty:** $0.1 \, \|\mathbf{v}\|^2$
 * 4. **Information reward:** $-\lambda_I \, \bar{H}$, where $\bar{H}$ is the
 *    mean per-beam visibility-weighted entropy along 8 omnidirectional rays.
 */
struct FSMICost
{
  const OccupancyGrid * map;   ///< Device pointer to 3D occupancy grid.
  float lambda_info;           ///< Information gain weight $\lambda_I$.
  float sensor_range;          ///< Maximum sensor range (m).
  float3 goal;                 ///< Target goal position.
  float lambda_goal;           ///< Goal attraction weight $\lambda_g$.

  /**
   * @brief Running cost at timestep $t$.
   *
   * @param x       Current state.
   * @param u       Current control.
   * @param u_prev  Previous control (unused).
   * @param t       Timestep index.
   * @return        Scalar running cost.
   */
  __device__ float compute(const float * x, const float * u,
                           const float * /*u_prev*/, int t) const
  {
    // 1. Control regularisation
    float cost = 0.0f;
    cost += 0.1f * (u[0] * u[0] + u[1] * u[1]);

    // 2. Goal attraction
    float dx_g = x[0] - goal.x;
    float dy_g = x[1] - goal.y;
    float dist_sq = dx_g * dx_g + dy_g * dy_g;
    cost += lambda_goal * dist_sq;

    // 3. Velocity penalty
    cost += 0.1f * (x[2] * x[2] + x[3] * x[3]);

    // 4. Omnidirectional entropy-based information reward
    if (map == nullptr || lambda_info <= 0.0f) {return cost;}

    float total_info = 0.0f;
    int num_beams = 8;

    for (int b = 0; b < num_beams; ++b) {
      float angle = b * (2.0f * 3.14159265f / (float)num_beams);
      float dx = cosf(angle) * map->resolution;
      float dy = sinf(angle) * map->resolution;

      float current_vis = 1.0f;
      float beam_info = 0.0f;
      float cx = x[0];
      float cy = x[1];
      float cz = 0.0f;

      int num_steps = (int)(sensor_range / map->resolution);
      for (int k = 0; k < num_steps; ++k) {
        cx += dx;
        cy += dy;
        float p = map->get_probability(make_float3(cx, cy, cz));
        p = (p < 0.001f) ? 0.001f : ((p > 0.999f) ? 0.999f : p);
        float entropy = -p * logf(p) - (1.0f - p) * logf(1.0f - p);
        beam_info += current_vis * entropy;
        current_vis *= (1.0f - p);
        if (current_vis < 0.01f) {break;}
      }
      total_info += beam_info;
    }

    cost -= lambda_info * (total_info / (float)num_beams);
    return cost;
  }

  /**
   * @brief Terminal cost (strong goal attraction).
   *
   * @param x  Terminal state.
   * @return   $100 \, \|\mathbf{p} - \mathbf{g}\|^2$.
   */
  __device__ float terminal_cost(const float * x) const
  {
    float dx_g = x[0] - goal.x;
    float dy_g = x[1] - goal.y;
    return 100.0f * (dx_g * dx_g + dy_g * dy_g);
  }
};

}  // namespace mppi

#endif  // FSMI_COST_CUH
