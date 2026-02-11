#ifndef FSMI_COST_CUH
#define FSMI_COST_CUH

#include <cuda_runtime.h>
#include "mppi/core/mppi_common.cuh"
#include "mppi/core/map.cuh"

namespace mppi
{

struct FSMICost
{
  const OccupancyGrid * map;   // Device pointer to map struct
  float lambda_info;          // Information gain weight
  float sensor_range;         // Max range in meters
  float3 goal;                // Target goal
  float lambda_goal;          // Goal weight

    // Additional parameters for dynamics/collision can be added.

  __device__ float compute(const float * x, const float * u, int t) const
  {
        // 1. Motion Cost (Regularization)
    float cost = 0.0f;
    cost += 0.1f * (u[0] * u[0] + u[1] * u[1]);

        // 2. Goal Cost (Stage)
    float dx_g = x[0] - goal.x;
    float dy_g = x[1] - goal.y;
    float dist_sq = dx_g * dx_g + dy_g * dy_g;
    cost += lambda_goal * dist_sq;

        // 3. Velocity Penalty (Stability)
    cost += 0.1f * (x[2] * x[2] + x[3] * x[3]);

        // 4. Information Reward (Omnidirectional FSMI)
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

  __device__ float terminal_cost(const float * x) const
  {
    float dx_g = x[0] - goal.x;
    float dy_g = x[1] - goal.y;
    return 100.0f * (dx_g * dx_g + dy_g * dy_g);   // Strong terminal attraction
  }
};

} // namespace mppi

#endif // FSMI_COST_CUH
