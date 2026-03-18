/**
 * @file bspline.cuh
 * @brief GPU-accelerated cubic B-spline evaluation for MPPI control sampling.
 *
 * Provides De Boor's algorithm on the GPU and a batch expansion kernel that
 * converts $K$ sets of $n$ control points into $K \times H$ per-timestep
 * control sequences.
 *
 * ## Uniform knot vector
 *
 * For $n+1$ control points ($n = n\_cp - 1$), degree $p = 3$ (cubic):
 *
 * $$
 *   u_i = (i - p) \cdot \Delta_k, \qquad i = 0, \ldots, m
 * $$
 *
 * where $m = n + p + 1$ and $\Delta_k$ is the knot span. The valid
 * parameter range is $[u_p, u_{m-p}] = [0, (n\_cp - p) \cdot \Delta_k]$.
 *
 * ## Batch expansion
 *
 * The `bspline_expand_kernel` maps:
 *
 * $$
 *   \text{control\_points}[K \times n\_cp \times n_u]
 *   \;\longrightarrow\;
 *   \text{u\_out}[K \times H \times n_u]
 * $$
 *
 * One thread per $(k, t)$ pair evaluates De Boor at $t \cdot \Delta t$.
 *
 * @see FUEL's `NonUniformBspline` for CPU reference implementation.
 */

#ifndef MPPI_BSPLINE_CUH
#define MPPI_BSPLINE_CUH

#include <cuda_runtime.h>
#include <cmath>

namespace mppi {

/// Maximum number of control points supported (compile-time upper bound).
static constexpr int BSPLINE_MAX_CP = 32;

/// B-spline degree (cubic).
static constexpr int BSPLINE_DEGREE = 3;

/**
 * @brief Compute uniform knot value at index i.
 *
 * For a uniform knot vector with degree p:
 *   u_i = (i - p) * knot_span
 *
 * @param i          Knot index (0 to m).
 * @param p          B-spline degree.
 * @param knot_span  Uniform spacing between knots.
 * @return           Knot value u_i.
 */
__device__ __host__ inline float uniform_knot(int i, int p, float knot_span)
{
  return static_cast<float>(i - p) * knot_span;
}

/**
 * @brief Evaluate a cubic B-spline at parameter u using De Boor's algorithm.
 *
 * For degree p=3, finds the knot span [u_k, u_{k+1}) containing u,
 * then iteratively computes the point using p+1 = 4 control points.
 *
 * @param u           Parameter value in [0, duration].
 * @param cp          Control points array, row-major [n_cp x nu].
 * @param n_cp        Number of control points.
 * @param nu          Control dimension (e.g., 2 for [a, alpha]).
 * @param knot_span   Uniform knot spacing.
 * @param out         Output control vector [nu].
 */
__device__ void deboor_eval(
  float u,
  const float* cp,
  int n_cp,
  int nu,
  float knot_span,
  float* out)
{
  const int p = BSPLINE_DEGREE;
  const int n = n_cp - 1;       // last control point index
  const int m = n + p + 1;      // last knot index

  // Clamp u to valid range [u_p, u_{m-p}]
  float u_start = uniform_knot(p, p, knot_span);       // = 0
  float u_end = uniform_knot(m - p, p, knot_span);     // = (n_cp - p) * knot_span
  u = fminf(fmaxf(u, u_start), u_end - 1e-6f);

  // Find knot span index k such that u in [u_k, u_{k+1})
  int k = p;
  while (k < m - p - 1 && uniform_knot(k + 1, p, knot_span) <= u) {
    ++k;
  }

  // De Boor's algorithm: work with p+1 = 4 points
  // d[j] starts as cp[k-p+j] for j = 0..p
  float d[4 * 12];  // max p+1=4 points, max nu=12 dims each

  for (int j = 0; j <= p; ++j) {
    int cp_idx = k - p + j;
    // Clamp to valid range
    cp_idx = max(0, min(cp_idx, n));
    for (int dim = 0; dim < nu; ++dim) {
      d[j * nu + dim] = cp[cp_idx * nu + dim];
    }
  }

  // Triangular computation
  for (int r = 1; r <= p; ++r) {
    for (int j = p; j >= r; --j) {
      int i_knot = j + k - p;  // corresponds to i in the original algorithm
      float u_left = uniform_knot(i_knot, p, knot_span);
      float u_right = uniform_knot(i_knot + 1 + p - r, p, knot_span);
      float denom = u_right - u_left;
      float alpha = (denom > 1e-8f) ? (u - u_left) / denom : 0.0f;

      for (int dim = 0; dim < nu; ++dim) {
        d[j * nu + dim] = (1.0f - alpha) * d[(j - 1) * nu + dim]
                        + alpha * d[j * nu + dim];
      }
    }
  }

  // Result is in d[p]
  for (int dim = 0; dim < nu; ++dim) {
    out[dim] = d[p * nu + dim];
  }
}

/**
 * @brief Host-side De Boor evaluation (mirrors device version).
 */
inline void deboor_eval_host(
  float u,
  const float* cp,
  int n_cp,
  int nu,
  float knot_span,
  float* out)
{
  const int p = BSPLINE_DEGREE;
  const int n = n_cp - 1;
  const int m = n + p + 1;

  float u_start = static_cast<float>(0);
  float u_end = static_cast<float>(n_cp - p) * knot_span;
  u = std::fmin(std::fmax(u, u_start), u_end - 1e-6f);

  int k = p;
  while (k < m - p - 1 && static_cast<float>(k + 1 - p) * knot_span <= u) {
    ++k;
  }

  float d[4 * 12];
  for (int j = 0; j <= p; ++j) {
    int cp_idx = std::max(0, std::min(k - p + j, n));
    for (int dim = 0; dim < nu; ++dim) {
      d[j * nu + dim] = cp[cp_idx * nu + dim];
    }
  }

  for (int r = 1; r <= p; ++r) {
    for (int j = p; j >= r; --j) {
      int i_knot = j + k - p;
      float u_left = static_cast<float>(i_knot - p) * knot_span;
      float u_right = static_cast<float>(i_knot + 1 + p - r - p) * knot_span;
      float denom = u_right - u_left;
      float alpha = (denom > 1e-8f) ? (u - u_left) / denom : 0.0f;

      for (int dim = 0; dim < nu; ++dim) {
        d[j * nu + dim] = (1.0f - alpha) * d[(j - 1) * nu + dim]
                        + alpha * d[j * nu + dim];
      }
    }
  }

  for (int dim = 0; dim < nu; ++dim) {
    out[dim] = d[p * nu + dim];
  }
}

/**
 * @brief CUDA kernel: expand B-spline control points to per-timestep controls.
 *
 * Each thread handles one (sample k, timestep t) pair.
 *
 * @param[in]  d_control_points  Control points [K x n_cp x nu] (device).
 * @param[out] d_u_out           Output controls [K x H x nu] (device).
 * @param[in]  K                 Number of samples.
 * @param[in]  H                 Prediction horizon.
 * @param[in]  n_cp              Number of control points per sample.
 * @param[in]  nu                Control dimension.
 * @param[in]  dt                Integration timestep.
 * @param[in]  knot_span         Uniform knot spacing.
 */
__global__ void bspline_expand_kernel(
  const float* __restrict__ d_control_points,
  float* __restrict__ d_u_out,
  int K,
  int H,
  int n_cp,
  int nu,
  float dt,
  float knot_span)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * H;
  if (tid >= total) return;

  int k = tid / H;
  int t = tid % H;

  // Parameter value for this timestep
  float u = static_cast<float>(t) * dt;

  // Pointer to this sample's control points
  const float* cp = d_control_points + k * n_cp * nu;

  // Output location
  float* out = d_u_out + k * H * nu + t * nu;

  // Evaluate B-spline
  deboor_eval(u, cp, n_cp, nu, knot_span, out);
}

/**
 * @brief B-spline expansion configuration.
 *
 * Precomputes the knot span from the horizon and number of control points,
 * and provides host/device expansion utilities.
 */
struct BSplineConfig
{
  int n_cp = 8;           ///< Number of control points.
  float knot_span = 0.4f; ///< Uniform knot spacing (computed from H, dt, n_cp).

  /**
   * @brief Compute knot span from horizon parameters.
   *
   * The total B-spline duration must equal H * dt:
   *   duration = (n_cp - p) * knot_span = H * dt
   *   knot_span = H * dt / (n_cp - p)
   *
   * @param H   Prediction horizon.
   * @param dt  Integration timestep.
   */
  void compute_knot_span(int H, float dt)
  {
    int n_segments = n_cp - BSPLINE_DEGREE;
    if (n_segments <= 0) n_segments = 1;
    knot_span = (static_cast<float>(H) * dt) / static_cast<float>(n_segments);
  }

  /**
   * @brief Get the total B-spline duration.
   */
  float duration() const
  {
    return static_cast<float>(n_cp - BSPLINE_DEGREE) * knot_span;
  }

  /**
   * @brief Launch the batch expansion kernel.
   *
   * @param d_control_points  Device buffer [K x n_cp x nu].
   * @param d_u_out           Device buffer [K x H x nu].
   * @param K                 Number of samples.
   * @param H                 Prediction horizon.
   * @param nu                Control dimension.
   * @param dt                Integration timestep.
   */
  void expand(const float* d_control_points, float* d_u_out,
              int K, int H, int nu, float dt) const
  {
    int total = K * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bspline_expand_kernel<<<blocks, threads>>>(
      d_control_points, d_u_out, K, H, n_cp, nu, dt, knot_span);
    cudaDeviceSynchronize();
  }

  /**
   * @brief Host-side expansion of a single set of control points.
   *
   * @param cp      Control points [n_cp x nu] (host).
   * @param u_out   Output controls [H x nu] (host).
   * @param H       Prediction horizon.
   * @param nu      Control dimension.
   * @param dt      Integration timestep.
   */
  void expand_host(const float* cp, float* u_out,
                   int H, int nu, float dt) const
  {
    for (int t = 0; t < H; ++t) {
      float u = static_cast<float>(t) * dt;
      deboor_eval_host(u, cp, n_cp, nu, knot_span, u_out + t * nu);
    }
  }
};

}  // namespace mppi

#endif  // MPPI_BSPLINE_CUH
