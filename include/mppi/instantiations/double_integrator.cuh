/**
 * @file double_integrator.cuh
 * @brief 2D double-integrator dynamics and quadratic cost for MPPI benchmarking.
 *
 * ## State vector ($n_x = 4$)
 *
 * | Index | Symbol    | Description    |
 * |-------|-----------|----------------|
 * | 0     | $p_x$    | X position     |
 * | 1     | $p_y$    | Y position     |
 * | 2     | $v_x$    | X velocity     |
 * | 3     | $v_y$    | Y velocity     |
 *
 * ## Control ($n_u = 2$)
 *
 * | Index | Symbol | Description      |
 * |-------|--------|------------------|
 * | 0     | $a_x$  | X acceleration   |
 * | 1     | $a_y$  | Y acceleration   |
 *
 * Euler integration: $\mathbf{v} \mathrel{+}= \mathbf{a} \, \Delta t$,
 * $\mathbf{p} \mathrel{+}= \mathbf{v} \, \Delta t$.
 */

#ifndef MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_CUH
#define MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_CUH

#include <cuda_runtime.h>

namespace mppi {
namespace instantiations {

/**
 * @brief 2D double-integrator dynamics (Euler integration).
 */
struct DoubleIntegrator
{
  static constexpr int STATE_DIM = 4;     ///< $n_x = 4$.
  static constexpr int CONTROL_DIM = 2;   ///< $n_u = 2$.

  /**
   * @brief Euler integration step.
   *
   * @param state       Current state $[p_x, p_y, v_x, v_y]$.
   * @param u           Control $[a_x, a_y]$.
   * @param next_state  Output next state.
   * @param dt          Time step.
   */
  __host__ __device__ void step(
    const float * state, const float * u, float * next_state,
    float dt) const
  {
    next_state[0] = state[0] + state[2] * dt;
    next_state[1] = state[1] + state[3] * dt;
    next_state[2] = state[2] + u[0] * dt;
    next_state[3] = state[3] + u[1] * dt;
  }
};

/**
 * @brief Quadratic cost for regulation to the origin.
 *
 * Running cost: $\|\mathbf{x}\|^2 + \|\mathbf{u}\|^2$.
 * Terminal cost: $10 \|\mathbf{x}\|^2$.
 */
struct QuadraticCost
{
  /** @brief Running cost. */
  __host__ __device__ float compute(const float * state, const float * u,
                                    const float * /*u_prev*/, int t) const
  {
    float c = 0.0f;
    for(int i = 0; i < 4; ++i) {
      c += state[i] * state[i];
    }
    for(int i = 0; i < 2; ++i) {
      c += u[i] * u[i];
    }
    return c;
  }

  /** @brief Terminal cost ($10\times$ state penalty). */
  __host__ __device__ float terminal_cost(const float * state) const
  {
    float c = 0.0f;
    for(int i = 0; i < 4; ++i) {
      c += state[i] * state[i] * 10.0f;
    }
    return c;
  }
};

}   // namespace instantiations
}  // namespace mppi

#endif  // MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_CUH
