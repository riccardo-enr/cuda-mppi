/**
 * @file point_mass_3d.cuh
 * @brief 3D point-mass translational dynamics for MPPI (NED frame).
 *
 * ## State vector ($n_x = 6$)
 *
 * | Index | Symbol    | Description     |
 * |-------|-----------|-----------------|
 * | 0–2   | $p_x, p_y, p_z$ | Position (NED)  |
 * | 3–5   | $v_x, v_y, v_z$ | Velocity (NED)  |
 *
 * ## Control ($n_u = 3$)
 *
 * | Index | Symbol    | Description               |
 * |-------|-----------|---------------------------|
 * | 0–2   | $a_x, a_y, a_z$ | Commanded acceleration (NED) |
 *
 * ## Dynamics
 *
 * $\dot{\mathbf{v}} = \mathbf{a}_{\text{cmd}}$ (PX4 handles gravity compensation).
 * $\dot{\mathbf{p}} = \mathbf{v}$.
 * Hover control: $[0, 0, 0]$.
 * Euler integration.
 */

#ifndef MPPI_INSTANTIATIONS_POINT_MASS_3D_CUH
#define MPPI_INSTANTIATIONS_POINT_MASS_3D_CUH

#include <cuda_runtime.h>
#include <Eigen/Dense>

namespace mppi {
namespace instantiations {

/**
 * @brief 3D point-mass dynamics with per-axis acceleration bounds.
 */
struct PointMass3D
{
  static constexpr int STATE_DIM = 6;     ///< $n_x = 6$.
  static constexpr int CONTROL_DIM = 3;   ///< $n_u = 3$.

  float gravity = 9.81f;                  ///< Gravitational acceleration (m/s$^2$).
  float a_max[3] = {5.0f, 5.0f, 5.0f};   ///< Per-axis acceleration bounds (m/s$^2$).

  /**
   * @brief Euler integration step (device).
   *
   * Clamps controls to $[-a_{\max}, a_{\max}]$, then integrates:
   * $\mathbf{v} \mathrel{+}= \mathbf{a} \, \Delta t$,
   * $\mathbf{p} \mathrel{+}= \mathbf{v}_{\text{new}} \, \Delta t$.
   *
   * @param x       Current state $[p_x, p_y, p_z, v_x, v_y, v_z]$.
   * @param u_raw   Unclamped commanded acceleration.
   * @param x_next  Output next state.
   * @param dt      Time step.
   */
  __device__ void step(
    const float* x, const float* u_raw,
    float* x_next, float dt) const
  {
    float u[3];
    for (int i = 0; i < 3; ++i) {
      u[i] = fminf(fmaxf(u_raw[i], -a_max[i]), a_max[i]);
    }

    x_next[3] = x[3] + u[0] * dt;
    x_next[4] = x[4] + u[1] * dt;
    x_next[5] = x[5] + u[2] * dt;

    x_next[0] = x[0] + x_next[3] * dt;
    x_next[1] = x[1] + x_next[4] * dt;
    x_next[2] = x[2] + x_next[5] * dt;
  }

  /**
   * @brief Host-side Euler step using Eigen vectors.
   *
   * @param state   State vector (modified in-place).
   * @param action  Commanded acceleration.
   * @param dt      Time step.
   */
  void step_host(Eigen::VectorXf& state, const Eigen::VectorXf& action, float dt) const
  {
    float u[3];
    for (int i = 0; i < 3; ++i) {
      float v = action(i);
      u[i] = std::fmin(std::fmax(v, -a_max[i]), a_max[i]);
    }

    state(3) += u[0] * dt;
    state(4) += u[1] * dt;
    state(5) += u[2] * dt;

    state(0) += state(3) * dt;
    state(1) += state(4) * dt;
    state(2) += state(5) * dt;
  }
};

}   // namespace instantiations
}   // namespace mppi

#endif  // MPPI_INSTANTIATIONS_POINT_MASS_3D_CUH
