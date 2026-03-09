/**
 * @file inverted_pendulum.cuh
 * @brief Cart-pole (inverted pendulum) dynamics and cost for MPPI benchmarking.
 *
 * ## State vector ($n_x = 4$)
 *
 * | Index | Symbol         | Description              |
 * |-------|----------------|--------------------------|
 * | 0     | $x$            | Cart position            |
 * | 1     | $\theta$       | Pole angle (0 = upright) |
 * | 2     | $\dot{x}$      | Cart velocity            |
 * | 3     | $\dot{\theta}$ | Pole angular velocity    |
 *
 * ## Control ($n_u = 1$)
 *
 * | Index | Symbol | Description       |
 * |-------|--------|-------------------|
 * | 0     | $F$    | Horizontal force  |
 *
 * ## Dynamics
 *
 * Standard cart-pole equations (OpenAI Gym convention, $\theta = 0$ is upright):
 *
 * $$
 *   \ddot{\theta} = \frac{g \sin\theta - \cos\theta \cdot \text{temp}}
 *                  {l \left(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p}\right)}
 * $$
 *
 * where $\text{temp} = \frac{F + m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}$.
 *
 * Euler integration.
 */

#ifndef MPPI_INSTANTIATIONS_INVERTED_PENDULUM_CUH
#define MPPI_INSTANTIATIONS_INVERTED_PENDULUM_CUH

#include <cuda_runtime.h>
#include <math.h>

namespace mppi {
namespace instantiations {

/**
 * @brief Cart-pole dynamics (Euler integration).
 */
struct InvertedPendulum
{
  static constexpr int STATE_DIM = 4;     ///< $n_x = 4$.
  static constexpr int CONTROL_DIM = 1;   ///< $n_u = 1$.

  static constexpr float g = 9.81f;       ///< Gravitational acceleration (m/s$^2$).
  static constexpr float mc = 1.0f;       ///< Cart mass (kg).
  static constexpr float mp = 0.1f;       ///< Pole mass (kg).
  static constexpr float l = 0.5f;        ///< Half-pole length (m).
  static constexpr float dt_default = 0.01f; ///< Default time step (s).

  /**
   * @brief Euler integration step.
   *
   * @param state       Current state $[x, \theta, \dot{x}, \dot{\theta}]$.
   * @param u           Control $[F]$.
   * @param next_state  Output next state.
   * @param dt          Time step.
   */
  __host__ __device__ void step(
    const float * state, const float * u, float * next_state,
    float dt) const
  {
    float th = state[1];
    float x_dot = state[2];
    float th_dot = state[3];
    float f = u[0];

    float sin_th = sinf(th);
    float cos_th = cosf(th);
    float total_m = mc + mp;

    float temp = (f + mp * l * th_dot * th_dot * sin_th) / total_m;
    float th_acc = (g * sin_th - cos_th * temp) /
      (l * (4.0f / 3.0f - mp * cos_th * cos_th / total_m));
    float x_acc = temp - mp * l * th_acc * cos_th / total_m;

    next_state[0] = state[0] + x_dot * dt;
    next_state[1] = th + th_dot * dt;
    next_state[2] = x_dot + x_acc * dt;
    next_state[3] = th_dot + th_acc * dt;
  }
};

/**
 * @brief Quadratic cost for cart-pole stabilisation at $\theta = 0$ (upright).
 *
 * Running cost: $x^2 + 10\theta^2 + 0.1\dot{x}^2 + 0.1\dot{\theta}^2 + 0.001 F^2$.
 * Terminal cost: $5x^2 + 50\theta^2 + \dot{x}^2 + \dot{\theta}^2$.
 */
struct PendulumCost
{
  /** @brief Running cost (penalises deviation from upright). */
  __host__ __device__ float compute(const float * state, const float * u,
                                    const float * /*u_prev*/, int t) const
  {
    float x = state[0];
    float theta = state[1];
    float x_dot = state[2];
    float theta_dot = state[3];

    float c = 0.0f;
    c += 1.0f * x * x;
    c += 10.0f * theta * theta;
    c += 0.1f * x_dot * x_dot;
    c += 0.1f * theta_dot * theta_dot;
    c += 0.001f * u[0] * u[0];

    return c;
  }

  /** @brief Terminal cost (stronger penalties). */
  __host__ __device__ float terminal_cost(const float * state) const
  {
    float c = 0.0f;
    c += 5.0f * state[0] * state[0];
    c += 50.0f * state[1] * state[1];
    c += 1.0f * state[2] * state[2];
    c += 1.0f * state[3] * state[3];
    return c;
  }
};

}   // namespace instantiations
}  // namespace mppi

#endif  // MPPI_INSTANTIATIONS_INVERTED_PENDULUM_CUH
