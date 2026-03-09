/**
 * @file quadrotor.cuh
 * @brief 13D quadrotor dynamics model (NED world frame, FRD body frame).
 *
 * ## State vector ($n_x = 13$)
 *
 * | Index | Symbol           | Description                |
 * |-------|------------------|----------------------------|
 * | 0–2   | $p_x, p_y, p_z$ | Position (NED)             |
 * | 3–5   | $v_x, v_y, v_z$ | Velocity (NED)             |
 * | 6–9   | $q_w, q_x, q_y, q_z$ | Attitude quaternion   |
 * | 10–12 | $\omega_x, \omega_y, \omega_z$ | Body angular velocity |
 *
 * ## Control vector ($n_u = 4$)
 *
 * | Index | Symbol          | Description          | Bounds            |
 * |-------|-----------------|----------------------|-------------------|
 * | 0     | $T$             | Collective thrust    | $[0, 4g]$         |
 * | 1–3   | $\omega_{x,y,z}^{\text{cmd}}$ | Body rate commands | $[-10, 10]$ rad/s |
 *
 * ## Dynamics
 *
 * - Thrust acts along the body $-z$ axis (FRD convention).
 * - Gravity is $[0, 0, mg]$ in NED (positive $z$ is down).
 * - Quaternion derivative: $\dot{q} = \frac{1}{2} q \otimes [0, \boldsymbol{\omega}]$.
 * - Angular velocity tracks commands via first-order lag: $\dot{\omega} = (\omega^{\text{cmd}} - \omega) / \tau$.
 * - Integration: 4th-order Runge–Kutta (RK4).
 *
 * Ported from `jax_mppi/dynamics/quadrotor.py`.
 *
 * @see `IMPPIController`, `InformativeCost` for typical usage.
 */

#ifndef MPPI_QUADROTOR_CUH
#define MPPI_QUADROTOR_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <Eigen/Dense>

namespace mppi {
namespace instantiations {

/**
 * @brief 13D quadrotor dynamics with RK4 integration.
 */
struct QuadrotorDynamics
{
  static constexpr int STATE_DIM = 13;    ///< $n_x = 13$.
  static constexpr int CONTROL_DIM = 4;   ///< $n_u = 4$.

  float mass = 1.0f;           ///< Vehicle mass (kg).
  float gravity = 9.81f;       ///< Gravitational acceleration (m/s$^2$).
  float tau_omega = 0.05f;     ///< Body-rate time constant $\tau$ (s).

  float u_min[4] = {0.0f, -10.0f, -10.0f, -10.0f};   ///< Control lower bounds.
  float u_max[4] = {39.24f, 10.0f, 10.0f, 10.0f};    ///< Control upper bounds ($T_{\max} = 4g$).

  /**
   * @brief Rotate a vector from body (FRD) to world (NED) frame.
   *
   * Applies the rotation matrix derived from quaternion $q$.
   *
   * @param q    Quaternion $[q_w, q_x, q_y, q_z]$.
   * @param v    Input vector in body frame.
   * @param out  Output vector in world frame.
   */
  __device__ __host__ static void quat_rotate(
    const float * q, const float * v, float * out
  )
  {
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];

    float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    float r01 = 2.0f * (qx * qy - qw * qz);
    float r02 = 2.0f * (qx * qz + qw * qy);
    float r10 = 2.0f * (qx * qy + qw * qz);
    float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    float r12 = 2.0f * (qy * qz - qw * qx);
    float r20 = 2.0f * (qx * qz - qw * qy);
    float r21 = 2.0f * (qy * qz + qw * qx);
    float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);

    out[0] = r00 * v[0] + r01 * v[1] + r02 * v[2];
    out[1] = r10 * v[0] + r11 * v[1] + r12 * v[2];
    out[2] = r20 * v[0] + r21 * v[1] + r22 * v[2];
  }

  /**
   * @brief Compute the continuous-time state derivative $\dot{\mathbf{x}}$.
   *
   * @param x   Current state $\in \mathbb{R}^{13}$.
   * @param u   Clamped control $\in \mathbb{R}^{4}$.
   * @param dx  Output state derivative $\in \mathbb{R}^{13}$.
   */
  __device__ __host__ void state_deriv(
    const float * x, const float * u, float * dx
  ) const
  {
    // Position derivative = velocity
    dx[0] = x[3]; dx[1] = x[4]; dx[2] = x[5];

    // Thrust in body frame: [0, 0, -T] (FRD: thrust opposes body z)
    float body_thrust[3] = {0.0f, 0.0f, -u[0]};
    float world_thrust[3];
    quat_rotate(x + 6, body_thrust, world_thrust);

    // Gravity: [0, 0, m*g] (NED: positive Z is down)
    dx[3] = world_thrust[0] / mass;
    dx[4] = world_thrust[1] / mass;
    dx[5] = (mass * gravity + world_thrust[2]) / mass;

    // Quaternion derivative: q_dot = 0.5 * q (x) [0, omega]
    float qw = x[6], qx = x[7], qy = x[8], qz = x[9];
    float wx = x[10], wy = x[11], wz = x[12];

    dx[6] = 0.5f * (-wx * qx - wy * qy - wz * qz);
    dx[7] = 0.5f * ( wx * qw + wz * qy - wy * qz);
    dx[8] = 0.5f * ( wy * qw - wz * qx + wx * qz);
    dx[9] = 0.5f * ( wz * qw + wy * qx - wx * qy);

    // Angular velocity: first-order tracking
    dx[10] = (u[1] - x[10]) / tau_omega;
    dx[11] = (u[2] - x[11]) / tau_omega;
    dx[12] = (u[3] - x[12]) / tau_omega;
  }

  /**
   * @brief RK4 integration step (called by rollout kernel).
   *
   * Clamps controls to `[u_min, u_max]`, integrates with RK4,
   * and normalises the quaternion.
   *
   * @param x       Current state $\in \mathbb{R}^{13}$.
   * @param u_raw   Raw (unclamped) control $\in \mathbb{R}^{4}$.
   * @param x_next  Output next state $\in \mathbb{R}^{13}$.
   * @param dt      Integration time step.
   */
  __device__ void step(
    const float * x, const float * u_raw,
    float * x_next, float dt) const
  {
    // Clamp controls
    float u[4];
    for (int i = 0; i < 4; ++i) {
      float v = u_raw[i];
      u[i] = fminf(fmaxf(v, u_min[i]), u_max[i]);
    }

    float k1[13], k2[13], k3[13], k4[13];
    float tmp[13];

    state_deriv(x, u, k1);

    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + 0.5f * dt * k1[i];
    }
    state_deriv(tmp, u, k2);

    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + 0.5f * dt * k2[i];
    }
    state_deriv(tmp, u, k3);

    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + dt * k3[i];
    }
    state_deriv(tmp, u, k4);

    for (int i = 0; i < 13; ++i) {
      x_next[i] = x[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    // Normalize quaternion
    float qnorm = sqrtf(
            x_next[6] * x_next[6] + x_next[7] * x_next[7] +
            x_next[8] * x_next[8] + x_next[9] * x_next[9]
      ) + 1e-8f;
    x_next[6] /= qnorm;
    x_next[7] /= qnorm;
    x_next[8] /= qnorm;
    x_next[9] /= qnorm;
  }

  /**
   * @brief Host-side RK4 step using Eigen vectors (for simulation loop).
   *
   * @param state   State vector (modified in-place).
   * @param action  Control vector.
   * @param dt      Integration time step.
   */
  void step_host(Eigen::VectorXf & state, const Eigen::VectorXf & action, float dt) const
  {
    float x[13], u_raw[4], x_next[13];
    for (int i = 0; i < 13; ++i) {
      x[i] = state(i);
    }
    for (int i = 0; i < 4; ++i) {
      u_raw[i] = action(i);
    }

    float u[4];
    for (int i = 0; i < 4; ++i) {
      u[i] = fminf(fmaxf(u_raw[i], u_min[i]), u_max[i]);
    }

    float k1[13], k2[13], k3[13], k4[13], tmp[13];

    state_deriv(x, u, k1);
    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + 0.5f * dt * k1[i];
    }
    state_deriv(tmp, u, k2);
    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + 0.5f * dt * k2[i];
    }
    state_deriv(tmp, u, k3);
    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + dt * k3[i];
    }
    state_deriv(tmp, u, k4);

    for (int i = 0; i < 13; ++i) {
      x_next[i] = x[i] + (dt / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    float qn = sqrtf(
            x_next[6] * x_next[6] + x_next[7] * x_next[7] +
            x_next[8] * x_next[8] + x_next[9] * x_next[9]
      ) + 1e-8f;
    for (int i = 6; i < 10; ++i) {
      x_next[i] /= qn;
    }

    for (int i = 0; i < 13; ++i) {
      state(i) = x_next[i];
    }
  }
};

}   // namespace instantiations
}  // namespace mppi

#endif  // MPPI_QUADROTOR_CUH
