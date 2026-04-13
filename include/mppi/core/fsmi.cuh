/**
 * @file fsmi.cuh
 * @brief Fast Shannon Mutual Information (FSMI) for GPU-accelerated information gain.
 *
 * Implements the FSMI algorithm from Zhang et al. (2020) for computing the
 * expected mutual information between a range sensor measurement and an
 * occupancy grid map. Provides both the full $O(n^2)$ banded formulation
 * and a fast $O(n)$ uniform approximation suitable for real-time cost
 * evaluation inside MPPI rollouts.
 *
 * Also includes:
 * - An `InfoField` struct that precomputes a 2D potential field of
 *   max-over-yaw FSMI values for strategic guidance.
 * - A FOV-based grid update kernel for simulated sensor observations.
 *
 * @see Zhang et al., "FSMI: Fast computation of Shannon Mutual Information
 *      for information-theoretic mapping", ICRA 2020.
 */

#ifndef MPPI_FSMI_CUH
#define MPPI_FSMI_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "mppi/core/map.cuh"
#include "mppi/utils/cuda_utils.cuh"

namespace mppi {

// ===========================================================================
// Configuration Structs
// ===========================================================================

/**
 * @brief Configuration for full FSMI computation (planner-level).
 *
 * Controls the sensor model, ray-casting resolution, and Gaussian
 * measurement noise used in the banded $G_{kj}$ formulation (Eq. 22).
 */
struct FSMIConfig {
  /// @name Planner-level parameters
  /// @{
  float info_threshold = 20.0f;   ///< Minimum MI to consider a pose informative.
  float ref_speed = 2.0f;         ///< Reference speed for trajectory generation (m/s).
  float info_weight = 10.0f;      ///< Weight on information gain in planner cost.
  float motion_weight = 1.0f;     ///< Weight on motion cost in planner.
  float dist_weight = 0.5f;       ///< Weight on distance-to-goal in planner.
  /// @}

  /// @name Goal
  /// @{
  float3 goal_pos = {9.0f, 5.0f, -2.0f};  ///< Target goal position (NED).
  /// @}

  /// @name Sensor model (Zhang et al. 2020)
  /// @{
  float fov_rad = 1.57f;          ///< Field-of-view half-angle (rad), default 90 deg.
  int   num_beams = 16;           ///< Number of beams spanning the FOV.
  float max_range = 5.0f;         ///< Maximum sensor range (m).
  float ray_step = 0.1f;          ///< Ray-marching step size (m).
  /// @}

  /// @name Sensor noise model
  /// @{
  float sigma_range = 0.15f;      ///< Range measurement std dev $\sigma_r$ (m).
  /// @}

  /// @name Inverse sensor model (log-odds)
  /// @{
  float inv_sensor_model_occ = 0.85f;   ///< $\ln(p_{\text{occ}} / (1 - p_{\text{occ}}))$.
  float inv_sensor_model_emp = -0.4f;   ///< $\ln(p_{\text{emp}} / (1 - p_{\text{emp}}))$.
  /// @}

  /// @name Gaussian truncation
  /// @{
  float gaussian_truncation_sigma = 3.0f;  ///< $G_{kj}$ bandwidth in units of $\sigma_r$.
  /// @}

  /// @name Trajectory-level parameters
  /// @{
  int   trajectory_subsample_rate = 5;   ///< Subsample rate along trajectory for IG.
  float trajectory_ig_decay = 0.7f;      ///< Exponential decay for cumulative IG.
  /// @}
};

/**
 * @brief Configuration for the uniform (fast) FSMI approximation.
 *
 * Uses the diagonal approximation $G_{kj} \approx \delta(k - j)$,
 * reducing per-beam complexity from $O(n^2)$ to $O(n)$. Suitable for
 * real-time evaluation inside MPPI rollout kernels.
 */
struct UniformFSMIConfig {
  float fov_rad = 1.57f;          ///< FOV half-angle (rad).
  int   num_beams = 6;            ///< Number of beams (fewer than full FSMI).
  float max_range = 2.5f;         ///< Local sensing range (m).
  float ray_step = 0.2f;          ///< Coarser ray step (m).

  float inv_sensor_model_occ = 0.85f;  ///< Log-odds for occupied cells.
  float inv_sensor_model_emp = -0.4f;  ///< Log-odds for empty cells.

  float info_weight = 5.0f;       ///< Cost weight for uniform FSMI reward.
};

/**
 * @brief Configuration for the precomputed information potential field.
 *
 * The `InfoField` is a 2D grid centred on the UAV that stores
 * $\max_\psi \text{FSMI}(x, y, \psi)$ at each cell. Updated periodically
 * on the GPU and sampled via bilinear interpolation in the cost function.
 */
struct InfoFieldConfig {
  float field_res = 0.5f;         ///< Metres per field cell.
  float field_extent = 5.0f;      ///< Half-width of the local field (m).
  int   n_yaw = 8;                ///< Number of yaw angles to maximise over.
  int   field_update_interval = 10; ///< Update every N control steps.
  float lambda_info = 5.0f;       ///< Cost weight for field lookup.
  float lambda_local = 10.0f;     ///< Cost weight for uniform FSMI.
  float ref_speed = 2.0f;         ///< Reference speed for trajectory generation (m/s).
  int   ref_horizon = 40;         ///< Reference trajectory horizon (steps).
  float target_weight = 1.0f;     ///< Weight on reference trajectory tracking.
};

// ===========================================================================
// Device Helpers
// ===========================================================================

/**
 * @brief Approximate Gaussian CDF via Abramowitz & Stegun rational approximation.
 *
 * Maximum absolute error $\approx 1.5 \times 10^{-7}$.
 *
 * @param x  Standard normal quantile.
 * @return   $\Phi(x) = P(Z \leq x)$ for $Z \sim \mathcal{N}(0,1)$.
 */
__device__ inline float norm_cdf(float x)
{
  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027f;
  const float a5 = 1.061405429f;
  const float p = 0.3275911f;

  float sign = (x < 0.0f) ? -1.0f : 1.0f;
  x = fabsf(x);

  float t = 1.0f / (1.0f + p * x);
  float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
    expf(-x * x * 0.5f);

  return 0.5f * (1.0f + sign * y);
}

/**
 * @brief F-score from Zhang et al. (2020), Eq. 9.
 *
 * $$
 *   f(r, \delta) = \ln\!\frac{r + 1}{r + 1/\delta} - \frac{\ln \delta}{r \delta + 1}
 * $$
 *
 * where $r = p / (1 - p)$ is the prior odds and
 * $\delta = \exp(\text{inv\_sensor\_model})$.
 *
 * @param r      Prior odds ratio (clamped to $\geq 10^{-4}$).
 * @param delta  $\exp(\text{inverse sensor model value})$.
 * @return       Scalar information contribution.
 */
__device__ inline float f_score(float r, float delta)
{
  r = fmaxf(r, 1e-4f);
  float term1 = logf((r + 1.0f) / (r + 1.0f / delta));
  float term2 = logf(delta) / (r * delta + 1.0f);
  return term1 - term2;
}

/// Maximum cells along a single beam (compile-time upper bound).
/// Full FSMI: $5.0 / 0.1 = 50$; uniform: $2.5 / 0.2 = 12$.
static constexpr int FSMI_MAX_CELLS = 64;

// ===========================================================================
// Full FSMI (per-beam, O(n^2) banded)
// ===========================================================================

/**
 * @brief Compute full FSMI for a single beam — $O(n^2)$ with banded $G_{kj}$.
 *
 * Implements Theorem 1, Algorithm 2 ($P(e_j)$), Algorithm 3 ($C_k$),
 * and Eq. 22 ($G_{kj}$) from Zhang et al. (2020).
 *
 * The mutual information for one beam is:
 *
 * $$
 *   I = \sum_j \sum_k P(e_j) \, C_k \, G_{kj}
 * $$
 *
 * where:
 * - $P(e_j) = o_j \prod_{l < j} (1 - o_l)$ is the event probability (Alg. 2)
 * - $C_k = f_{\text{occ}}(k) + \sum_{i < k} f_{\text{emp}}(i)$ (Alg. 3)
 * - $G_{kj} = \Phi\!\bigl(\frac{l_{k+\frac{1}{2}} - \mu_j}{\sigma}\bigr)
 *           - \Phi\!\bigl(\frac{l_{k-\frac{1}{2}} - \mu_j}{\sigma}\bigr)$ (Eq. 22)
 *
 * @param cell_probs  Occupancy probabilities along the ray ($N$ values).
 * @param cell_dists  Distances from the sensor along the ray ($N$ values).
 * @param N           Number of cells along this beam.
 * @param cfg         FSMI configuration (sensor noise, truncation).
 * @return            Beam mutual information (nats).
 */
__device__ float compute_beam_fsmi(
  const float * cell_probs,
  const float * cell_dists,
  int N,
  const FSMIConfig & cfg
)
{
  if (N <= 0 || N > FSMI_MAX_CELLS) {return 0.0f;}

  float P_e[FSMI_MAX_CELLS];
  float C_k[FSMI_MAX_CELLS];

    // === Algorithm 2: P(e_j) = o_j * prod_{l<j} (1 - o_l) ===
  float cum_not_occ = 1.0f;
  for (int j = 0; j < N; ++j) {
    P_e[j] = cell_probs[j] * cum_not_occ;
    cum_not_occ *= (1.0f - cell_probs[j]);
  }

    // === Algorithm 3: C_k = f_occ[k] + sum_{i<k} f_emp[i] ===
  float delta_occ = expf(cfg.inv_sensor_model_occ);
  float delta_emp = expf(cfg.inv_sensor_model_emp);

  float cum_f_emp = 0.0f;
  for (int k = 0; k < N; ++k) {
    float odds = cell_probs[k] / (1.0f - cell_probs[k] + 1e-6f);
    float focc = f_score(odds, delta_occ);
    float femp = f_score(odds, delta_emp);
    C_k[k] = focc + cum_f_emp;
    cum_f_emp += femp;
  }

    // === Eq. 22: G_kj with Gaussian truncation ===
  float sigma = cfg.sigma_range;
  float half_s = cfg.ray_step * 0.5f;
  int trunc_r = (int)(cfg.gaussian_truncation_sigma * sigma / cfg.ray_step + 0.5f);
  if (trunc_r < 1) {trunc_r = 1;}

    // Accumulate MI = sum_j sum_k P_e[j] * C_k[k] * G_kj
  float mi = 0.0f;
  for (int j = 0; j < N; ++j) {
    if (P_e[j] < 1e-8f) {continue;}    // skip negligible terms
    float mu_j = cell_dists[j];

    int k_lo = (j - trunc_r < 0) ? 0 : (j - trunc_r);
    int k_hi = (j + trunc_r >= N) ? (N - 1) : (j + trunc_r);

    for (int k = k_lo; k <= k_hi; ++k) {
      float l_k_plus = cell_dists[k] + half_s;
      float l_k_minus = cell_dists[k] - half_s;
      float z_hi = (l_k_plus - mu_j) / sigma;
      float z_lo = (l_k_minus - mu_j) / sigma;
      float G_kj = norm_cdf(z_hi) - norm_cdf(z_lo);
      mi += P_e[j] * C_k[k] * G_kj;
    }
  }

  return mi;
}

// ===========================================================================
// Uniform FSMI (per-beam, O(n) approximation)
// ===========================================================================

/**
 * @brief Compute uniform FSMI for a single beam — $O(n)$ approximation.
 *
 * Assumes perfect range measurement ($G_{kj} \approx \delta(k - j)$),
 * collapsing the double sum to:
 *
 * $$
 *   I \approx \sum_j P(e_j) \, C_j
 * $$
 *
 * @param cell_probs  Occupancy probabilities along the ray ($N$ values).
 * @param N           Number of cells along this beam.
 * @param cfg         Uniform FSMI configuration.
 * @return            Beam mutual information (nats, approximate).
 */
__device__ float compute_beam_uniform_fsmi(
  const float * cell_probs,
  int N,
  const UniformFSMIConfig & cfg
)
{
  if (N <= 0 || N > FSMI_MAX_CELLS) {return 0.0f;}

  float delta_occ = expf(cfg.inv_sensor_model_occ);
  float delta_emp = expf(cfg.inv_sensor_model_emp);

  float cum_not_occ = 1.0f;
  float cum_f_emp = 0.0f;
  float mi = 0.0f;

  for (int j = 0; j < N; ++j) {
    float p = cell_probs[j];
    float odds = p / (1.0f - p + 1e-6f);
    odds = fmaxf(odds, 1e-4f);

        // P(e_j)
    float P_e = p * cum_not_occ;

        // C_j
    float focc = f_score(odds, delta_occ);
    float femp = f_score(odds, delta_emp);
    float C_j = focc + cum_f_emp;

    mi += P_e * C_j;

    cum_not_occ *= (1.0f - p);
    cum_f_emp += femp;
  }

  return mi;
}

// ===========================================================================
// Pose-level FSMI (all beams)
// ===========================================================================

/**
 * @brief Compute full FSMI at a single 2D pose (all beams within FOV).
 *
 * Casts `num_beams` rays uniformly across the FOV centred at `yaw`,
 * and sums per-beam FSMI values.
 *
 * @param grid  2D occupancy grid (device-accessible).
 * @param pos   UAV position in world frame.
 * @param yaw   Heading angle (rad).
 * @param cfg   Full FSMI configuration.
 * @return      Total mutual information across all beams (nats).
 */
__device__ float compute_fsmi_at_pose(
  const OccupancyGrid2D & grid,
  float2 pos,
  float yaw,
  const FSMIConfig & cfg
)
{
  int num_cells = (int)(cfg.max_range / cfg.ray_step);
  if (num_cells > FSMI_MAX_CELLS) {num_cells = FSMI_MAX_CELLS;}

  float cell_probs[FSMI_MAX_CELLS];
  float cell_dists[FSMI_MAX_CELLS];

  float total_mi = 0.0f;

  for (int b = 0; b < cfg.num_beams; ++b) {
        // Uniform angles across FOV
    float angle = yaw - cfg.fov_rad * 0.5f +
      cfg.fov_rad * (float)b / fmaxf((float)(cfg.num_beams - 1), 1.0f);

    float dx = cosf(angle);
    float dy = sinf(angle);

        // Cast ray
    for (int c = 0; c < num_cells; ++c) {
      float dist = (c + 0.5f) * cfg.ray_step;        // distance to cell centre
      float2 wp = {pos.x + dist * dx, pos.y + dist * dy};

      int2 gi = grid.world_to_grid(wp);
      int idx = grid.get_index(gi.x, gi.y);
      float p = (idx >= 0) ? grid.data[idx] : 0.5f;

      cell_probs[c] = p;
      cell_dists[c] = dist;
    }

    total_mi += compute_beam_fsmi(cell_probs, cell_dists, num_cells, cfg);
  }

  return total_mi;
}

/**
 * @brief Compute uniform FSMI at a single 2D pose (all beams within FOV).
 *
 * Lightweight version of `compute_fsmi_at_pose` using the $O(n)$
 * uniform approximation. Designed for use inside MPPI rollout costs.
 *
 * @param grid  2D occupancy grid (device-accessible).
 * @param pos   UAV position in world frame.
 * @param yaw   Heading angle (rad).
 * @param cfg   Uniform FSMI configuration.
 * @return      Total approximate mutual information (nats).
 */
__device__ float compute_uniform_fsmi_at_pose(
  const OccupancyGrid2D & grid,
  float2 pos,
  float yaw,
  const UniformFSMIConfig & cfg
)
{
  int num_cells = (int)(cfg.max_range / cfg.ray_step);
  if (num_cells > FSMI_MAX_CELLS) {num_cells = FSMI_MAX_CELLS;}

  float cell_probs[FSMI_MAX_CELLS];

  float total_mi = 0.0f;

  for (int b = 0; b < cfg.num_beams; ++b) {
    float angle = yaw - cfg.fov_rad * 0.5f +
      cfg.fov_rad * (float)b / fmaxf((float)(cfg.num_beams - 1), 1.0f);

    float dx = cosf(angle);
    float dy = sinf(angle);

    for (int c = 0; c < num_cells; ++c) {
      float dist = (c + 0.5f) * cfg.ray_step;
      float2 wp = {pos.x + dist * dx, pos.y + dist * dy};

      int2 gi = grid.world_to_grid(wp);
      int idx = grid.get_index(gi.x, gi.y);
      float p = (idx >= 0) ? grid.data[idx] : 0.5f;

      cell_probs[c] = p;
    }

    total_mi += compute_beam_uniform_fsmi(cell_probs, num_cells, cfg);
  }

  return total_mi;
}

// ===========================================================================
// Information Field
// ===========================================================================

/**
 * @brief CUDA kernel to compute the information potential field.
 *
 * One thread per $(i_x, i_y)$ cell. For each cell, evaluates FSMI at
 * `n_yaw` uniformly spaced heading angles and stores the maximum:
 *
 * $$
 *   F[i_x, i_y] = \max_{\psi \in [0, 2\pi)} \text{FSMI}(x, y, \psi)
 * $$
 *
 * @param grid          2D occupancy grid.
 * @param field_output  Output field $[N_x \times N_y]$ (device).
 * @param field_origin  World coordinates of field cell $(0, 0)$.
 * @param field_res     Field resolution (m/cell).
 * @param Nx            Field width in cells.
 * @param Ny            Field height in cells.
 * @param n_yaw         Number of yaw angles to evaluate.
 * @param fsmi_cfg      Full FSMI configuration.
 */
__global__ void compute_info_field_kernel(
  const OccupancyGrid2D grid,
  float *       field_output,
  float2       field_origin,
  float        field_res,
  int          Nx,
  int          Ny,
  int          n_yaw,
  FSMIConfig   fsmi_cfg
)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= Nx || iy >= Ny) {return;}

  float2 pos = {
    field_origin.x + ((float)ix + 0.5f) * field_res,
    field_origin.y + ((float)iy + 0.5f) * field_res
  };

  float best_mi = -1e30f;
  for (int y = 0; y < n_yaw; ++y) {
    float yaw = 2.0f * 3.14159265f * (float)y / (float)n_yaw;
    float mi = compute_fsmi_at_pose(grid, pos, yaw, fsmi_cfg);
    if (mi > best_mi) {best_mi = mi;}
  }

  field_output[ix * Ny + iy] = best_mi;
}

/**
 * @brief Host-side manager for the 2D information potential field.
 *
 * Maintains a device-allocated grid of precomputed FSMI values centred
 * on the UAV. The field is recomputed periodically (e.g. at 5 Hz) by
 * launching `compute_info_field_kernel`, and sampled inside the MPPI
 * cost function via bilinear interpolation (`sample()`).
 */
struct InfoField
{
  float * d_field = nullptr;    ///< Device pointer to field data $[N_x \times N_y]$.
  float2 origin = {0.0f, 0.0f}; ///< World coordinates of cell $(0, 0)$.
  float  res = 0.5f;            ///< Field resolution (m/cell).
  int    Nx = 0;                ///< Field width in cells.
  int    Ny = 0;                ///< Field height in cells.

  /** @brief Allocate (or reallocate) device memory for the field. */
  void allocate(int nx, int ny)
  {
    Nx = nx;
    Ny = ny;
    if (d_field) {HANDLE_ERROR(cudaFree(d_field));}
    HANDLE_ERROR(cudaMalloc(&d_field, Nx * Ny * sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_field, 0, Nx * Ny * sizeof(float)));
  }

  /** @brief Free device memory. */
  void free()
  {
    if (d_field) {cudaFree(d_field); d_field = nullptr;}
  }

  /**
   * @brief Recompute the field centred on the UAV position.
   *
   * @param grid      Current 2D occupancy grid.
   * @param uav_pos   UAV position in world frame (field will be centred here).
   * @param ifc       InfoField configuration.
   * @param fsmi_cfg  Full FSMI configuration for per-cell evaluation.
   */
  void compute(
    const OccupancyGrid2D & grid,
    float2 uav_pos,
    const InfoFieldConfig & ifc,
    const FSMIConfig & fsmi_cfg
  )
  {
    int nx = (int)(2.0f * ifc.field_extent / ifc.field_res);
    int ny = nx;
    if (Nx != nx || Ny != ny) {allocate(nx, ny);}

    origin.x = uav_pos.x - ifc.field_extent;
    origin.y = uav_pos.y - ifc.field_extent;
    res = ifc.field_res;

    dim3 block(16, 16);
    dim3 grid_dim(
      (Nx + block.x - 1) / block.x,
      (Ny + block.y - 1) / block.y);

    compute_info_field_kernel << < grid_dim, block >> > (
      grid, d_field, origin, res, Nx, Ny, ifc.n_yaw, fsmi_cfg
      );
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  /**
   * @brief Download the field to host memory.
   * @param h_field  Host buffer of size $N_x \times N_y$.
   */
  void download(float * h_field) const
  {
    if (d_field && h_field) {
      HANDLE_ERROR(cudaMemcpy(h_field, d_field, Nx * Ny * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
  }

  /**
   * @brief Sample the field at a world position via bilinear interpolation.
   *
   * Designed for use inside `__device__` cost functions. Clamps to field
   * boundaries and returns 0 if the field is not allocated.
   *
   * @param world_pos  Query position in world coordinates.
   * @return           Interpolated FSMI value.
   */
  __device__ float sample(float2 world_pos) const
  {
    if (!d_field || Nx <= 0 || Ny <= 0) {return 0.0f;}

        // Continuous field-cell coordinates
    float fx = (world_pos.x - origin.x) / res - 0.5f;
    float fy = (world_pos.y - origin.y) / res - 0.5f;

    // Return 0 for queries outside the field — no reward beyond the
    // computed extent. Clamping to edge cells caused boundary attraction.
    if (fx < -0.5f || fx > (float)(Nx - 1) + 0.5f ||
        fy < -0.5f || fy > (float)(Ny - 1) + 0.5f) {
      return 0.0f;
    }

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float sx = fx - (float)x0;
    float sy = fy - (float)y0;

        // Clamp (for bilinear interpolation at valid edges)
    auto clamp = [] (int v, int lo, int hi) {
        return (v < lo) ? lo : ((v > hi) ? hi : v);
      };
    int x0c = clamp(x0, 0, Nx - 1);
    int x1c = clamp(x0 + 1, 0, Nx - 1);
    int y0c = clamp(y0, 0, Ny - 1);
    int y1c = clamp(y0 + 1, 0, Ny - 1);

    float v00 = d_field[x0c * Ny + y0c];
    float v10 = d_field[x1c * Ny + y0c];
    float v01 = d_field[x0c * Ny + y1c];
    float v11 = d_field[x1c * Ny + y1c];

    return (1.0f - sx) * (1.0f - sy) * v00 +
           sx * (1.0f - sy) * v10 +
           (1.0f - sx) * sy * v01 +
           sx * sy * v11;
  }
};

// ===========================================================================
// FOV Grid Update Kernel
// ===========================================================================

/**
 * @brief CUDA kernel to update a 2D occupancy grid along sensor FOV rays.
 *
 * One thread per ray. Marches from `uav_pos` outward, marking cells as
 * free (`free_update`) until an obstacle ($p \geq$ `occ_threshold`) is
 * hit, which is marked as occupied (`occ_update`) and terminates the ray.
 *
 * @param grid_data      Occupancy grid probability data (device, row-major).
 * @param width          Grid width in cells.
 * @param height         Grid height in cells.
 * @param origin         World coordinates of cell $(0, 0)$.
 * @param resolution     Grid resolution (m/cell).
 * @param uav_pos        UAV position in world frame.
 * @param yaw            UAV heading (rad).
 * @param fov_rad        Field-of-view half-angle (rad).
 * @param max_range      Maximum ray length (m).
 * @param n_rays         Number of rays to cast.
 * @param ray_step       Step size along each ray (m).
 * @param free_update    Probability to write for free cells (e.g. 0.01).
 * @param occ_update     Probability to write for obstacle cells (e.g. 0.99).
 * @param occ_threshold  Threshold above which a cell is considered occupied.
 */
__global__ void fov_grid_update_kernel(
  float * grid_data,
  int    width,
  int    height,
  float2 origin,
  float  resolution,
  float2 uav_pos,
  float  yaw,
  float  fov_rad,
  float  max_range,
  int    n_rays,
  float  ray_step,
  float  free_update,
  float  occ_update,
  float  occ_threshold
)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_rays) {return;}

  float angle = yaw - fov_rad * 0.5f +
    fov_rad * (float)r / fmaxf((float)(n_rays - 1), 1.0f);
  float dx = cosf(angle);
  float dy = sinf(angle);

  int n_steps = (int)(max_range / ray_step);
  bool blocked = false;

  for (int s = 0; s < n_steps && !blocked; ++s) {
    float dist = ((float)s + 0.5f) * ray_step;
    float wx = uav_pos.x + dist * dx;
    float wy = uav_pos.y + dist * dy;

    int cx = (int)((wx - origin.x) / resolution);
    int cy = (int)((wy - origin.y) / resolution);
    if (cx < 0 || cx >= width || cy < 0 || cy >= height) {break;}

    int idx = cy * width + cx;
    float p = grid_data[idx];

    if (p >= occ_threshold) {
            // Known obstacle — mark as occupied and stop ray
      grid_data[idx] = occ_update;
      blocked = true;
    } else {
            // Free/unknown — mark as known-free
      grid_data[idx] = free_update;
    }
  }
}

/**
 * @brief Host wrapper for FOV-based grid update.
 *
 * Launches `fov_grid_update_kernel` and synchronizes.
 *
 * @param grid           2D occupancy grid to update (device data).
 * @param uav_pos        UAV position in world frame.
 * @param yaw            UAV heading (rad).
 * @param fov_rad        FOV half-angle (rad), default 90 deg.
 * @param max_range      Max ray range (m), default 2.5.
 * @param n_rays         Number of rays, default 64.
 * @param ray_step       Step size (m), default 0.1.
 * @param free_update    Free-cell probability, default 0.01.
 * @param occ_update     Obstacle probability, default 0.99.
 * @param occ_threshold  Occupied threshold, default 0.7.
 */
inline void fov_grid_update(
  OccupancyGrid2D & grid,
  float2 uav_pos,
  float yaw,
  float fov_rad = 1.57f,
  float max_range = 2.5f,
  int   n_rays = 64,
  float ray_step = 0.1f,
  float free_update = 0.01f,
  float occ_update = 0.99f,
  float occ_threshold = 0.7f
)
{
  dim3 block(256);
  dim3 grid_dim((n_rays + block.x - 1) / block.x);

  fov_grid_update_kernel << < grid_dim, block >> > (
    grid.data, grid.dims.x, grid.dims.y,
    grid.origin, grid.resolution,
    uav_pos, yaw, fov_rad, max_range, n_rays, ray_step,
    free_update, occ_update, occ_threshold
    );
  cudaDeviceSynchronize();
}

}  // namespace mppi

#endif  // MPPI_FSMI_CUH
