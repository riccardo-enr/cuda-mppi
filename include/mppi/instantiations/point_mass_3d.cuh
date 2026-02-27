#ifndef MPPI_INSTANTIATIONS_POINT_MASS_3D_CUH
#define MPPI_INSTANTIATIONS_POINT_MASS_3D_CUH

#include <cuda_runtime.h>
#include <Eigen/Dense>

namespace mppi
{
namespace instantiations
{

// ---------------------------------------------------------------------------
// 3D Point-Mass Translational Dynamics (NED frame)
//
// State (6):   [px, py, pz, vx, vy, vz]
// Control (3): [ax, ay, az]   (commanded acceleration in NED)
//
// v_dot = a_cmd + [0, 0, g]   (gravity: +g in NED Z-down)
// p_dot = v
//
// Hover control: [0, 0, -g]  → net acceleration = 0
// Euler integration (linear dynamics)
// ---------------------------------------------------------------------------
struct PointMass3D
{
  static constexpr int STATE_DIM = 6;
  static constexpr int CONTROL_DIM = 3;

  float gravity = 9.81f;
  float a_max[3] = {5.0f, 5.0f, 5.0f};  // per-axis acceleration bounds

  __device__ void step(
    const float* x, const float* u_raw,
    float* x_next, float dt) const
  {
    // Clamp controls
    float u[3];
    for (int i = 0; i < 3; ++i) {
      u[i] = fminf(fmaxf(u_raw[i], -a_max[i]), a_max[i]);
    }

    // v_dot = a_cmd + [0, 0, g]
    float ax = u[0];
    float ay = u[1];
    float az = u[2] + gravity;

    // Euler integration: v += a * dt, p += v * dt
    x_next[3] = x[3] + ax * dt;
    x_next[4] = x[4] + ay * dt;
    x_next[5] = x[5] + az * dt;

    x_next[0] = x[0] + x_next[3] * dt;
    x_next[1] = x[1] + x_next[4] * dt;
    x_next[2] = x[2] + x_next[5] * dt;
  }

  // Host-side step using Eigen (for trajectory rollout)
  void step_host(Eigen::VectorXf& state, const Eigen::VectorXf& action, float dt) const
  {
    float u[3];
    for (int i = 0; i < 3; ++i) {
      float v = action(i);
      u[i] = std::fmin(std::fmax(v, -a_max[i]), a_max[i]);
    }

    float ax = u[0];
    float ay = u[1];
    float az = u[2] + gravity;

    state(3) += ax * dt;
    state(4) += ay * dt;
    state(5) += az * dt;

    state(0) += state(3) * dt;
    state(1) += state(4) * dt;
    state(2) += state(5) * dt;
  }
};

}   // namespace instantiations
}   // namespace mppi

#endif // MPPI_INSTANTIATIONS_POINT_MASS_3D_CUH
