/**
 * @file unicycle.cuh
 * @brief Unicycle dynamics with velocity/omega state for UGV MPPI.
 *
 * ## State vector ($n_x = 5$)
 *
 * | Index | Symbol      | Description           |
 * |-------|-------------|-----------------------|
 * | 0     | $x$         | Position x (NED)      |
 * | 1     | $y$         | Position y (NED)      |
 * | 2     | $\theta$    | Heading angle (rad)   |
 * | 3     | $v$         | Forward velocity (m/s)|
 * | 4     | $\omega$    | Angular velocity (rad/s)|
 *
 * ## Control ($n_u = 2$)
 *
 * | Index | Symbol      | Description                     |
 * |-------|-------------|---------------------------------|
 * | 0     | $a$         | Linear acceleration (m/s^2)     |
 * | 1     | $\alpha$    | Angular acceleration (rad/s^2)  |
 *
 * ## Dynamics
 *
 * $\dot{x} = v \cos\theta$
 * $\dot{y} = v \sin\theta$
 * $\dot{\theta} = \omega$
 * $\dot{v} = (a_{\text{cmd}} - c_d v) / m$
 * $\dot{\omega} = \alpha_{\text{cmd}}$
 *
 * RK4 integration, velocity/omega clamping, heading wrapping.
 */

#ifndef MPPI_INSTANTIATIONS_UNICYCLE_CUH
#define MPPI_INSTANTIATIONS_UNICYCLE_CUH

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <cmath>

namespace mppi {
namespace instantiations {

struct UnicycleDynamics
{
  static constexpr int STATE_DIM = 5;
  static constexpr int CONTROL_DIM = 2;

  float v_max = 2.0f;
  float omega_max = 2.0f;
  float a_max = 4.0f;
  float alpha_max = 3.0f;
  float drag = 0.1f;
  float mass = 5.0f;

  /**
   * @brief Compute state derivatives for RK4.
   */
  __device__ __host__ void derivatives(
    const float* x, const float* u_clamp,
    float* dx) const
  {
    float v = x[3];
    float theta = x[2];
    float omega = x[4];

    dx[0] = v * cosf(theta);
    dx[1] = v * sinf(theta);
    dx[2] = omega;
    dx[3] = (u_clamp[0] - drag * v) / mass;
    dx[4] = u_clamp[1];
  }

  /**
   * @brief Wrap angle to [-pi, pi].
   */
  __device__ __host__ static float wrap_angle(float a)
  {
    a = fmodf(a + 3.14159265f, 2.0f * 3.14159265f);
    if (a < 0.0f) a += 2.0f * 3.14159265f;
    return a - 3.14159265f;
  }

  /**
   * @brief RK4 integration step (device).
   */
  __device__ void step(
    const float* x, const float* u_raw,
    float* x_next, float dt) const
  {
    // Clamp controls
    float u[2];
    u[0] = fminf(fmaxf(u_raw[0], -a_max), a_max);
    u[1] = fminf(fmaxf(u_raw[1], -alpha_max), alpha_max);

    float k1[5], k2[5], k3[5], k4[5], x_tmp[5];

    // k1
    derivatives(x, u, k1);

    // k2
    for (int i = 0; i < 5; ++i) x_tmp[i] = x[i] + 0.5f * dt * k1[i];
    derivatives(x_tmp, u, k2);

    // k3
    for (int i = 0; i < 5; ++i) x_tmp[i] = x[i] + 0.5f * dt * k2[i];
    derivatives(x_tmp, u, k3);

    // k4
    for (int i = 0; i < 5; ++i) x_tmp[i] = x[i] + dt * k3[i];
    derivatives(x_tmp, u, k4);

    // Combine
    for (int i = 0; i < 5; ++i) {
      x_next[i] = x[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    // Wrap heading
    x_next[2] = wrap_angle(x_next[2]);

    // Clamp velocity and omega
    x_next[3] = fminf(fmaxf(x_next[3], -v_max), v_max);
    x_next[4] = fminf(fmaxf(x_next[4], -omega_max), omega_max);
  }

  /**
   * @brief Host-side RK4 step using Eigen vectors.
   */
  void step_host(Eigen::VectorXf& state, const Eigen::VectorXf& action, float dt) const
  {
    float x[5], u[2], x_next[5];
    for (int i = 0; i < 5; ++i) x[i] = state(i);

    // Clamp controls
    u[0] = std::fmin(std::fmax(action(0), -a_max), a_max);
    u[1] = std::fmin(std::fmax(action(1), -alpha_max), alpha_max);

    float k1[5], k2[5], k3[5], k4[5], x_tmp[5];

    derivatives(x, u, k1);
    for (int i = 0; i < 5; ++i) x_tmp[i] = x[i] + 0.5f * dt * k1[i];
    derivatives(x_tmp, u, k2);
    for (int i = 0; i < 5; ++i) x_tmp[i] = x[i] + 0.5f * dt * k2[i];
    derivatives(x_tmp, u, k3);
    for (int i = 0; i < 5; ++i) x_tmp[i] = x[i] + dt * k3[i];
    derivatives(x_tmp, u, k4);

    for (int i = 0; i < 5; ++i) {
      x_next[i] = x[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    x_next[2] = wrap_angle(x_next[2]);
    x_next[3] = std::fmin(std::fmax(x_next[3], -v_max), v_max);
    x_next[4] = std::fmin(std::fmax(x_next[4], -omega_max), omega_max);

    for (int i = 0; i < 5; ++i) state(i) = x_next[i];
  }
};

}  // namespace instantiations
}  // namespace mppi

#endif  // MPPI_INSTANTIATIONS_UNICYCLE_CUH
