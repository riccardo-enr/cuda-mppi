/**
 * @file double_integrator_3d.cuh
 * @brief 3D double-integrator dynamics for outer-loop MPPI planning.
 *
 * Models a point mass in NED with acceleration control — the natural
 * dynamics for an MPPI planner that sits above an SO(3) attitude
 * controller.
 *
 * ## State vector ($n_x = 6$)
 *
 * | Index | Symbol    | Description      |
 * |-------|-----------|------------------|
 * | 0     | $p_x$    | North position   |
 * | 1     | $p_y$    | East position    |
 * | 2     | $p_z$    | Down position    |
 * | 3     | $v_x$    | North velocity   |
 * | 4     | $v_y$    | East velocity    |
 * | 5     | $v_z$    | Down velocity    |
 *
 * ## Control ($n_u = 3$)
 *
 * | Index | Symbol | Description          |
 * |-------|--------|----------------------|
 * | 0     | $a_x$  | North acceleration   |
 * | 1     | $a_y$  | East acceleration    |
 * | 2     | $a_z$  | Down acceleration    |
 *
 * Gravity is **not** included — the SO(3) inner loop handles it.
 * The control represents the desired inertial acceleration setpoint.
 *
 * Integration: Euler ($\dot{p} = v$, $\dot{v} = a$).
 *
 * Acceleration is clamped to `[a_min, a_max]` per axis.
 */

#ifndef MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_3D_CUH
#define MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_3D_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <Eigen/Dense>

namespace mppi {
namespace instantiations {

struct DoubleIntegrator3D
{
  static constexpr int STATE_DIM = 6;
  static constexpr int CONTROL_DIM = 3;

  float a_max[3] = {5.0f, 5.0f, 5.0f};   ///< Per-axis max acceleration (m/s²). Allow asymmetric limits so lateral can be bounded harder than vertical.

  /**
   * @brief Euler integration step with acceleration clamping.
   */
  __host__ __device__ void step(
    const float * x, const float * u_raw,
    float * x_next, float dt) const
  {
    float u[3];
    for (int i = 0; i < 3; ++i) {
      u[i] = fminf(fmaxf(u_raw[i], -a_max[i]), a_max[i]);
    }

    x_next[0] = x[0] + x[3] * dt;
    x_next[1] = x[1] + x[4] * dt;
    x_next[2] = x[2] + x[5] * dt;
    x_next[3] = x[3] + u[0] * dt;
    x_next[4] = x[4] + u[1] * dt;
    x_next[5] = x[5] + u[2] * dt;
  }

  /**
   * @brief Host-side step using Eigen vectors (for Python simulation loop).
   */
  void step_host(Eigen::VectorXf & state, const Eigen::VectorXf & action, float dt) const
  {
    float x[6], u[3], x_next[6];
    for (int i = 0; i < 6; ++i) { x[i] = state(i); }
    for (int i = 0; i < 3; ++i) { u[i] = action(i); }
    step(x, u, x_next, dt);
    for (int i = 0; i < 6; ++i) { state(i) = x_next[i]; }
  }
};

}  // namespace instantiations
}  // namespace mppi

#endif  // MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_3D_CUH
