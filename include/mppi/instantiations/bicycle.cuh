/**
 * @file bicycle.cuh
 * @brief Bicycle dynamics for UGV MPPI.
 *
 * ## State vector ($n_x = 4$)
 *
 * | Index | Symbol      | Description           |
 * |-------|-------------|-----------------------|
 * | 0     | $x$         | Position x (NED)      |
 * | 1     | $y$         | Position y (NED)      |
 * | 2     | $\theta$    | Heading angle (rad)   |
 * | 3     | $v$         | Forward velocity (m/s)|
 *
 * ## Control ($n_u = 2$)
 *
 * | Index | Symbol      | Description                       |
 * |-------|-------------|-----------------------------------|
 * | 0     | throttle    | Normalised throttle $\in [-1, 1]$ |
 * | 1     | steering    | Normalised steering $\in [-1, 1]$ |
 *
 * ## Dynamics
 *
 * $\dot{x} = v \cos\theta$
 * $\dot{y} = v \sin\theta$
 * $\dot{\theta} = \frac{v}{L} \tan(\delta)$, where $\delta =$ steering $\times \delta_{\max}$
 * $\dot{v} = \frac{F_{\text{throttle}} \times F_{\max} - c_d v}{m}$
 *
 * RK4 integration, velocity clamping, heading wrapping.
 * Controls are normalised $[-1, 1]$ and map directly to ActuatorMotors.
 */

#ifndef MPPI_INSTANTIATIONS_BICYCLE_CUH
#define MPPI_INSTANTIATIONS_BICYCLE_CUH

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <cmath>

namespace mppi {
namespace instantiations {

struct BicycleDynamics
{
  static constexpr int STATE_DIM = 4;
  static constexpr int CONTROL_DIM = 2;

  float wheelbase = 0.5f;
  float max_steering_angle = 0.6f;
  float max_speed = 2.0f;
  float mass = 5.0f;
  float drag = 0.1f;
  float max_throttle_force = 20.0f;

  __device__ __host__ void derivatives(
    const float* x, const float* u_clamp,
    float* dx) const
  {
    float v = x[3];
    float theta = x[2];
    float delta = u_clamp[1] * max_steering_angle;
    float F = u_clamp[0] * max_throttle_force;

    dx[0] = v * cosf(theta);
    dx[1] = v * sinf(theta);
    dx[2] = (v / wheelbase) * tanf(delta);
    dx[3] = (F - drag * v) / mass;
  }

  __device__ __host__ static float wrap_angle(float a)
  {
    a = fmodf(a + 3.14159265f, 2.0f * 3.14159265f);
    if (a < 0.0f) a += 2.0f * 3.14159265f;
    return a - 3.14159265f;
  }

  __device__ void step(
    const float* x, const float* u_raw,
    float* x_next, float dt) const
  {
    float u[2];
    u[0] = fminf(fmaxf(u_raw[0], -1.0f), 1.0f);
    u[1] = fminf(fmaxf(u_raw[1], -1.0f), 1.0f);

    float k1[4], k2[4], k3[4], k4[4], x_tmp[4];

    derivatives(x, u, k1);
    for (int i = 0; i < 4; ++i) x_tmp[i] = x[i] + 0.5f * dt * k1[i];
    derivatives(x_tmp, u, k2);
    for (int i = 0; i < 4; ++i) x_tmp[i] = x[i] + 0.5f * dt * k2[i];
    derivatives(x_tmp, u, k3);
    for (int i = 0; i < 4; ++i) x_tmp[i] = x[i] + dt * k3[i];
    derivatives(x_tmp, u, k4);

    for (int i = 0; i < 4; ++i) {
      x_next[i] = x[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    x_next[2] = wrap_angle(x_next[2]);
    x_next[3] = fminf(fmaxf(x_next[3], -max_speed), max_speed);
  }

  void step_host(Eigen::VectorXf& state, const Eigen::VectorXf& action, float dt) const
  {
    float x[4], u[2], x_next[4];
    for (int i = 0; i < 4; ++i) x[i] = state(i);

    u[0] = std::fmin(std::fmax(action(0), -1.0f), 1.0f);
    u[1] = std::fmin(std::fmax(action(1), -1.0f), 1.0f);

    float k1[4], k2[4], k3[4], k4[4], x_tmp[4];

    derivatives(x, u, k1);
    for (int i = 0; i < 4; ++i) x_tmp[i] = x[i] + 0.5f * dt * k1[i];
    derivatives(x_tmp, u, k2);
    for (int i = 0; i < 4; ++i) x_tmp[i] = x[i] + 0.5f * dt * k2[i];
    derivatives(x_tmp, u, k3);
    for (int i = 0; i < 4; ++i) x_tmp[i] = x[i] + dt * k3[i];
    derivatives(x_tmp, u, k4);

    for (int i = 0; i < 4; ++i) {
      x_next[i] = x[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    x_next[2] = wrap_angle(x_next[2]);
    x_next[3] = std::fmin(std::fmax(x_next[3], -max_speed), max_speed);

    for (int i = 0; i < 4; ++i) state(i) = x_next[i];
  }
};

}  // namespace instantiations
}  // namespace mppi

#endif  // MPPI_INSTANTIATIONS_BICYCLE_CUH
