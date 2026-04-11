/**
 * @file tracking_cost.cuh
 * @brief Quadratic trajectory tracking cost for quadrotor MPPI.
 *
 * Implements the standard MPPI cost from Enrico et al. (2025), Eq. 7:
 *
 * $$
 *   q(x_k, u_k) = \|p_k - p_{\text{ref},k}\|^2_{Q_p}
 *               + \|v_k - v_{\text{ref},k}\|^2_{Q_v}
 *               + d_q(q_k, q_{\text{ref},k}) \, Q_q
 *               + \|\omega_k - \omega_{\text{ref},k}\|^2_{Q_\omega}
 *               + \|u_k\|^2_R
 *               + \|\Delta u_k\|^2_{R_\Delta}
 * $$
 *
 * where $d_q = 1 - \langle q_1, q_2 \rangle^2$ is the quaternion distance.
 *
 * No grids, no FSMI, no occupancy maps — pure quadratic tracking.
 *
 * @tparam STATE_DIM  Typically 13 for quadrotor.
 * @tparam CONTROL_DIM  Typically 4 for quadrotor.
 */

#ifndef MPPI_TRACKING_COST_CUH
#define MPPI_TRACKING_COST_CUH

#include <cuda_runtime.h>
#include <cmath>

namespace mppi {
namespace instantiations {

/**
 * @brief Quadratic tracking cost for quadrotor trajectory following.
 *
 * State: $[p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z,
 *          \omega_x, \omega_y, \omega_z]$
 *
 * Control: $[T, \omega_x^{\text{cmd}}, \omega_y^{\text{cmd}},
 *            \omega_z^{\text{cmd}}]$
 */
struct TrackingCost
{
  /* Position tracking weights (per axis). */
  float Q_pos[3] = {5e3f, 5e3f, 5e3f};

  /* Velocity tracking weights (per axis). */
  float Q_vel[3] = {4e1f, 4e1f, 4e1f};

  /* Quaternion error weight (scalar, applied to $d_q$). */
  float Q_quat = 2e1f;

  /* Angular rate tracking weights (per axis). */
  float Q_omega[3] = {2e1f, 2e1f, 2e1f};

  /* Control cost weights. */
  float R[4] = {2e-2f, 2e-1f, 2e-1f, 2e-1f};

  /* Control rate cost weights ($\|\Delta u\|^2$). */
  float R_delta[4] = {1e-3f, 1e-2f, 1e-2f, 1e-2f};

  /* Terminal cost multiplier relative to stage cost. */
  float terminal_weight = 10.0f;

  /**
   * @brief Reference trajectory on device.
   *
   * Layout: `[horizon × 13]` — full state reference per timestep.
   * If null, reference defaults to hover at `hover_state`.
   */
  const float * ref_trajectory = nullptr;
  int ref_horizon = 0;

  /* Default hover state used when no reference is set.
   * Position and velocity = 0; identity quaternion; zero angular rate. */
  float hover_pos[3] = {0.0f, 0.0f, -2.5f};

  /**
   * @brief Compute the running cost at timestep $t$.
   */
  __device__ float compute(const float * x, const float * u,
                           const float * u_prev, int t) const
  {
    float cost = 0.0f;

    /* Determine reference state for this timestep. */
    float x_ref[13] = {0.0f};
    if (ref_trajectory != nullptr && ref_horizon > 0) {
      int ti = (t < ref_horizon) ? t : (ref_horizon - 1);
      const float * src = ref_trajectory + ti * 13;
      for (int i = 0; i < 13; ++i) { x_ref[i] = src[i]; }
    } else {
      /* Hover reference: position from hover_pos, identity quat. */
      x_ref[0] = hover_pos[0];
      x_ref[1] = hover_pos[1];
      x_ref[2] = hover_pos[2];
      x_ref[6] = 1.0f;  /* qw = 1 */
    }

    /* 1. Position error: ||p - p_ref||^2_Q */
    for (int i = 0; i < 3; ++i) {
      float e = x[i] - x_ref[i];
      cost += Q_pos[i] * e * e;
    }

    /* 2. Velocity error: ||v - v_ref||^2_Q */
    for (int i = 0; i < 3; ++i) {
      float e = x[3 + i] - x_ref[3 + i];
      cost += Q_vel[i] * e * e;
    }

    /* 3. Quaternion error: d_q = 1 - <q, q_ref>^2 */
    float dot = x[6] * x_ref[6] + x[7] * x_ref[7]
              + x[8] * x_ref[8] + x[9] * x_ref[9];
    float dq = 1.0f - dot * dot;
    cost += Q_quat * dq;

    /* 4. Angular rate error: ||ω - ω_ref||^2_Q */
    for (int i = 0; i < 3; ++i) {
      float e = x[10 + i] - x_ref[10 + i];
      cost += Q_omega[i] * e * e;
    }

    /* 5. Control cost: ||u||^2_R */
    for (int i = 0; i < 4; ++i) {
      cost += R[i] * u[i] * u[i];
    }

    /* 6. Control rate cost: ||Δu||^2_R_Δ */
    if (u_prev != nullptr) {
      for (int i = 0; i < 4; ++i) {
        float du = u[i] - u_prev[i];
        cost += R_delta[i] * du * du;
      }
    }

    return cost;
  }

  /**
   * @brief Terminal cost: position + velocity + attitude at end of horizon.
   */
  __device__ float terminal_cost(const float * x) const
  {
    float cost = 0.0f;

    float x_ref[13] = {0.0f};
    if (ref_trajectory != nullptr && ref_horizon > 0) {
      const float * src = ref_trajectory + (ref_horizon - 1) * 13;
      for (int i = 0; i < 13; ++i) { x_ref[i] = src[i]; }
    } else {
      x_ref[0] = hover_pos[0];
      x_ref[1] = hover_pos[1];
      x_ref[2] = hover_pos[2];
      x_ref[6] = 1.0f;
    }

    for (int i = 0; i < 3; ++i) {
      float e = x[i] - x_ref[i];
      cost += Q_pos[i] * e * e;
    }
    for (int i = 0; i < 3; ++i) {
      float e = x[3 + i] - x_ref[3 + i];
      cost += Q_vel[i] * e * e;
    }
    float dot = x[6] * x_ref[6] + x[7] * x_ref[7]
              + x[8] * x_ref[8] + x[9] * x_ref[9];
    cost += Q_quat * (1.0f - dot * dot);

    return terminal_weight * cost;
  }
};

}  // namespace instantiations
}  // namespace mppi

#endif  // MPPI_TRACKING_COST_CUH
