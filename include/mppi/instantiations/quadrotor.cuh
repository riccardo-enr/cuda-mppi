#ifndef MPPI_QUADROTOR_CUH
#define MPPI_QUADROTOR_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <Eigen/Dense>

namespace mppi
{
namespace instantiations
{

// ---------------------------------------------------------------------------
// 13D Quadrotor Dynamics (NED world frame, FRD body frame)
//
// State (13): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
// Control (4): [T, wx_cmd, wy_cmd, wz_cmd]
//
// Ported from jax_mppi/dynamics/quadrotor.py (RK4 integration)
// ---------------------------------------------------------------------------
struct QuadrotorDynamics
{
  static constexpr int STATE_DIM = 13;
  static constexpr int CONTROL_DIM = 4;

  float mass = 1.0f;
  float gravity = 9.81f;
  float tau_omega = 0.05f;

  float u_min[4] = {0.0f, -10.0f, -10.0f, -10.0f};
  float u_max[4] = {39.24f, 10.0f, 10.0f, 10.0f};     // 4*g

    // Quaternion to rotation matrix (body FRD → world NED), applied to a vector
  __device__ __host__ static void quat_rotate(
    const float * q, const float * v, float * out
  )
  {
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];

        // R * v (rotation matrix from quaternion)
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

    // Compute state derivative dx/dt
  __device__ __host__ void state_deriv(
    const float * x, const float * u, float * dx
  ) const
  {
        // Position derivative = velocity
    dx[0] = x[3]; dx[1] = x[4]; dx[2] = x[5];

        // Thrust in body frame: [0, 0, -T]
    float body_thrust[3] = {0.0f, 0.0f, -u[0]};
    float world_thrust[3];
    quat_rotate(x + 6, body_thrust, world_thrust);

        // Gravity: [0, 0, m*g] (NED: positive Z is down)
    dx[3] = world_thrust[0] / mass;
    dx[4] = world_thrust[1] / mass;
    dx[5] = (mass * gravity + world_thrust[2]) / mass;

        // Quaternion derivative: q_dot = 0.5 * q ⊗ [0, ω]
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

    // RK4 step (called by rollout kernel)
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

        // k1
    state_deriv(x, u, k1);

        // k2
    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + 0.5f * dt * k1[i];
    }
    state_deriv(tmp, u, k2);

        // k3
    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + 0.5f * dt * k2[i];
    }
    state_deriv(tmp, u, k3);

        // k4
    for (int i = 0; i < 13; ++i) {
      tmp[i] = x[i] + dt * k3[i];
    }
    state_deriv(tmp, u, k4);

        // Integrate
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

    // Host-side step using Eigen (for simulation loop)
  void step_host(Eigen::VectorXf & state, const Eigen::VectorXf & action, float dt) const
  {
    float x[13], u_raw[4], x_next[13];
    for (int i = 0; i < 13; ++i) {
      x[i] = state(i);
    }
    for (int i = 0; i < 4; ++i) {
      u_raw[i] = action(i);
    }

        // Clamp controls
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

        // Normalize quaternion
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
} // namespace mppi

#endif // MPPI_QUADROTOR_CUH
