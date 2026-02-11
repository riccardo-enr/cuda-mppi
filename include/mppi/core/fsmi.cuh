#ifndef MPPI_FSMI_CUH
#define MPPI_FSMI_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "mppi/core/map.cuh"

namespace mppi {

// ---------------------------------------------------------------------------
// Configuration structs (ported from Python FSMIConfig, UniformFSMIConfig,
// InfoFieldConfig)
// ---------------------------------------------------------------------------

struct FSMIConfig {
    // Planner-level parameters
    float info_threshold = 20.0f;
    float ref_speed       = 2.0f;
    float info_weight     = 10.0f;
    float motion_weight   = 1.0f;
    float dist_weight     = 0.5f;

    // Goal position (x, y, z)
    float3 goal_pos = {9.0f, 5.0f, -2.0f};

    // Sensor parameters (Zhang et al. 2020)
    float fov_rad   = 1.57f;   // 90 degrees
    int   num_beams = 16;
    float max_range = 5.0f;    // metres
    float ray_step  = 0.1f;    // 10 cm cells along ray

    // Sensor noise model
    float sigma_range = 0.15f; // Gaussian std dev (metres)

    // Inverse sensor model (log-odds)
    float inv_sensor_model_occ = 0.85f;  // log(0.7/0.3)
    float inv_sensor_model_emp = -0.4f;  // log(0.4/0.6)

    // Gaussian truncation for G_kj (bandwidth)
    float gaussian_truncation_sigma = 3.0f;

    // Trajectory-level IG parameters
    int   trajectory_subsample_rate = 5;
    float trajectory_ig_decay       = 0.7f;
};

struct UniformFSMIConfig {
    float fov_rad   = 1.57f;   // 90 degrees
    int   num_beams = 6;
    float max_range = 2.5f;    // local only (metres)
    float ray_step  = 0.2f;    // coarser resolution

    float inv_sensor_model_occ = 0.85f;
    float inv_sensor_model_emp = -0.4f;

    float info_weight = 5.0f;
};

struct InfoFieldConfig {
    float field_res      = 0.5f;  // metres per field cell
    float field_extent   = 5.0f;  // half-width of local workspace (m)
    int   n_yaw          = 8;
    int   field_update_interval = 10;
    float lambda_info    = 5.0f;  // field lookup cost weight
    float lambda_local   = 10.0f; // Uniform-FSMI cost weight
    float ref_speed      = 2.0f;
    int   ref_horizon    = 40;
    float target_weight  = 1.0f;
    float goal_weight    = 0.5f;
};

// ---------------------------------------------------------------------------
// Device helper: approximate Gaussian CDF via the Abramowitz & Stegun
// rational approximation (max error ~1.5e-7).
// ---------------------------------------------------------------------------
__device__ inline float norm_cdf(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;

    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    x = fabsf(x);

    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
              * expf(-x * x * 0.5f);

    return 0.5f * (1.0f + sign * y);
}

// ---------------------------------------------------------------------------
// f-score: Eq. 9 from Zhang et al. (2020)
//   f(r, δ) = log((r+1)/(r+1/δ)) − log(δ)/(r·δ+1)
// where r = odds = p/(1−p), δ = exp(inv_sensor_model)
// ---------------------------------------------------------------------------
__device__ inline float f_score(float r, float delta) {
    r = fmaxf(r, 1e-4f);
    float term1 = logf((r + 1.0f) / (r + 1.0f / delta));
    float term2 = logf(delta) / (r * delta + 1.0f);
    return term1 - term2;
}

// ---------------------------------------------------------------------------
// Maximum cells along a single beam (compile-time upper bound)
// max_range / ray_step: 5.0/0.1 = 50 for full FSMI, 2.5/0.2 = 12 for uniform
// ---------------------------------------------------------------------------
static constexpr int FSMI_MAX_CELLS = 64;

// ---------------------------------------------------------------------------
// Full FSMI for a single beam — O(n²) with banded G_kj
//
// Implements Theorem 1, Algorithm 2 (P_e), Algorithm 3 (C_k), Eq. 22 (G_kj)
//
// cell_probs: occupancy probabilities along the ray (N values)
// cell_dists: distances from sensor along the ray (N values)
// N:          number of cells along this beam
// cfg:        FSMIConfig with sensor noise and truncation parameters
// ---------------------------------------------------------------------------
__device__ float compute_beam_fsmi(
    const float* cell_probs,
    const float* cell_dists,
    int N,
    const FSMIConfig& cfg
) {
    if (N <= 0 || N > FSMI_MAX_CELLS) return 0.0f;

    float P_e[FSMI_MAX_CELLS];
    float C_k[FSMI_MAX_CELLS];

    // === Algorithm 2: P(e_j) = o_j · ∏_{l<j} (1 − o_l) ===
    float cum_not_occ = 1.0f;
    for (int j = 0; j < N; ++j) {
        P_e[j] = cell_probs[j] * cum_not_occ;
        cum_not_occ *= (1.0f - cell_probs[j]);
    }

    // === Algorithm 3: C_k = f_occ[k] + Σ_{i<k} f_emp[i] ===
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
    // G_kj = Φ((l_{k+½} − μ_j)/σ) − Φ((l_{k−½} − μ_j)/σ)
    // Truncate: |k−j| > trunc_radius → G_kj = 0
    float sigma  = cfg.sigma_range;
    float half_s = cfg.ray_step * 0.5f;
    int trunc_r  = (int)(cfg.gaussian_truncation_sigma * sigma / cfg.ray_step + 0.5f);
    if (trunc_r < 1) trunc_r = 1;

    // Accumulate MI = Σ_j Σ_k P_e[j] · C_k[k] · G_kj
    float mi = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (P_e[j] < 1e-8f) continue;  // skip negligible terms
        float mu_j = cell_dists[j];

        int k_lo = (j - trunc_r < 0) ? 0 : (j - trunc_r);
        int k_hi = (j + trunc_r >= N) ? (N - 1) : (j + trunc_r);

        for (int k = k_lo; k <= k_hi; ++k) {
            float l_k_plus  = cell_dists[k] + half_s;
            float l_k_minus = cell_dists[k] - half_s;
            float z_hi = (l_k_plus  - mu_j) / sigma;
            float z_lo = (l_k_minus - mu_j) / sigma;
            float G_kj = norm_cdf(z_hi) - norm_cdf(z_lo);
            mi += P_e[j] * C_k[k] * G_kj;
        }
    }

    return mi;
}

// ---------------------------------------------------------------------------
// Uniform-FSMI for a single beam — O(n) approximation
//
// MI ≈ Σ_j P(e_j) · C_j   (G_kj ≈ δ(k−j))
// ---------------------------------------------------------------------------
__device__ float compute_beam_uniform_fsmi(
    const float* cell_probs,
    int N,
    const UniformFSMIConfig& cfg
) {
    if (N <= 0 || N > FSMI_MAX_CELLS) return 0.0f;

    float delta_occ = expf(cfg.inv_sensor_model_occ);
    float delta_emp = expf(cfg.inv_sensor_model_emp);

    float cum_not_occ = 1.0f;
    float cum_f_emp   = 0.0f;
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
        float C_j  = focc + cum_f_emp;

        mi += P_e * C_j;

        cum_not_occ *= (1.0f - p);
        cum_f_emp   += femp;
    }

    return mi;
}

// ---------------------------------------------------------------------------
// Compute full FSMI at a single pose (all beams within FOV)
// ---------------------------------------------------------------------------
__device__ float compute_fsmi_at_pose(
    const OccupancyGrid2D& grid,
    float2 pos,
    float yaw,
    const FSMIConfig& cfg
) {
    int num_cells = (int)(cfg.max_range / cfg.ray_step);
    if (num_cells > FSMI_MAX_CELLS) num_cells = FSMI_MAX_CELLS;

    float cell_probs[FSMI_MAX_CELLS];
    float cell_dists[FSMI_MAX_CELLS];

    float total_mi = 0.0f;

    for (int b = 0; b < cfg.num_beams; ++b) {
        // Uniform angles across FOV
        float angle = yaw - cfg.fov_rad * 0.5f
                    + cfg.fov_rad * (float)b / fmaxf((float)(cfg.num_beams - 1), 1.0f);

        float dx = cosf(angle);
        float dy = sinf(angle);

        // Cast ray
        for (int c = 0; c < num_cells; ++c) {
            float dist = (c + 0.5f) * cfg.ray_step;  // distance to cell centre
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

// ---------------------------------------------------------------------------
// Compute Uniform-FSMI at a single pose (all beams within FOV)
// ---------------------------------------------------------------------------
__device__ float compute_uniform_fsmi_at_pose(
    const OccupancyGrid2D& grid,
    float2 pos,
    float yaw,
    const UniformFSMIConfig& cfg
) {
    int num_cells = (int)(cfg.max_range / cfg.ray_step);
    if (num_cells > FSMI_MAX_CELLS) num_cells = FSMI_MAX_CELLS;

    float cell_probs[FSMI_MAX_CELLS];

    float total_mi = 0.0f;

    for (int b = 0; b < cfg.num_beams; ++b) {
        float angle = yaw - cfg.fov_rad * 0.5f
                    + cfg.fov_rad * (float)b / fmaxf((float)(cfg.num_beams - 1), 1.0f);

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

// ---------------------------------------------------------------------------
// Kernel: compute info field at a coarse grid of (x,y) positions.
// One thread per (ix, iy) cell. Loops over n_yaw angles, stores max.
// ---------------------------------------------------------------------------
__global__ void compute_info_field_kernel(
    const OccupancyGrid2D grid,
    float*       field_output,   // (Nx * Ny)
    float2       field_origin,   // world coords of field[0,0]
    float        field_res,
    int          Nx,
    int          Ny,
    int          n_yaw,
    FSMIConfig   fsmi_cfg
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;

    float2 pos = {
        field_origin.x + ((float)ix + 0.5f) * field_res,
        field_origin.y + ((float)iy + 0.5f) * field_res
    };

    float best_mi = -1e30f;
    for (int y = 0; y < n_yaw; ++y) {
        float yaw = 2.0f * 3.14159265f * (float)y / (float)n_yaw;
        float mi = compute_fsmi_at_pose(grid, pos, yaw, fsmi_cfg);
        if (mi > best_mi) best_mi = mi;
    }

    field_output[ix * Ny + iy] = best_mi;
}

// ---------------------------------------------------------------------------
// InfoField: host-side manager for the information potential field.
// ---------------------------------------------------------------------------
struct InfoField {
    float* d_field = nullptr;  // device pointer (Nx * Ny)
    float2 origin  = {0.0f, 0.0f};
    float  res     = 0.5f;
    int    Nx      = 0;
    int    Ny      = 0;

    void allocate(int nx, int ny) {
        Nx = nx;
        Ny = ny;
        if (d_field) cudaFree(d_field);
        cudaMalloc(&d_field, Nx * Ny * sizeof(float));
        cudaMemset(d_field, 0, Nx * Ny * sizeof(float));
    }

    void free() {
        if (d_field) { cudaFree(d_field); d_field = nullptr; }
    }

    // Compute field centred on uav_pos
    void compute(
        const OccupancyGrid2D& grid,
        float2 uav_pos,
        const InfoFieldConfig& ifc,
        const FSMIConfig& fsmi_cfg
    ) {
        int nx = (int)(2.0f * ifc.field_extent / ifc.field_res);
        int ny = nx;
        if (Nx != nx || Ny != ny) allocate(nx, ny);

        origin.x = uav_pos.x - ifc.field_extent;
        origin.y = uav_pos.y - ifc.field_extent;
        res = ifc.field_res;

        dim3 block(16, 16);
        dim3 grid_dim((Nx + block.x - 1) / block.x,
                      (Ny + block.y - 1) / block.y);

        compute_info_field_kernel<<<grid_dim, block>>>(
            grid, d_field, origin, res, Nx, Ny, ifc.n_yaw, fsmi_cfg
        );
        cudaDeviceSynchronize();
    }

    // Download field to host
    void download(float* h_field) const {
        if (d_field && h_field) {
            cudaMemcpy(h_field, d_field, Nx * Ny * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }

    // Bilinear interpolation on device (for use in cost function)
    __device__ float sample(float2 world_pos) const {
        if (!d_field || Nx <= 0 || Ny <= 0) return 0.0f;

        // Continuous field-cell coordinates
        float fx = (world_pos.x - origin.x) / res - 0.5f;
        float fy = (world_pos.y - origin.y) / res - 0.5f;

        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        float sx = fx - (float)x0;
        float sy = fy - (float)y0;

        // Clamp
        auto clamp = [](int v, int lo, int hi) {
            return (v < lo) ? lo : ((v > hi) ? hi : v);
        };
        int x0c = clamp(x0,     0, Nx - 1);
        int x1c = clamp(x0 + 1, 0, Nx - 1);
        int y0c = clamp(y0,     0, Ny - 1);
        int y1c = clamp(y0 + 1, 0, Ny - 1);

        float v00 = d_field[x0c * Ny + y0c];
        float v10 = d_field[x1c * Ny + y0c];
        float v01 = d_field[x0c * Ny + y1c];
        float v11 = d_field[x1c * Ny + y1c];

        return (1.0f - sx) * (1.0f - sy) * v00
             +        sx  * (1.0f - sy) * v10
             + (1.0f - sx) *        sy  * v01
             +        sx  *        sy  * v11;
    }
};

// ===========================================================================
// FOV Grid Update Kernel
// ===========================================================================

// One thread per ray. March along the ray and update visible cells.
__global__ void fov_grid_update_kernel(
    float* grid_data,
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
    float  free_update,   // probability to write for free cells (e.g. 0.01)
    float  occ_update,    // probability to write for obstacles  (e.g. 0.99)
    float  occ_threshold  // threshold above which a cell is an obstacle (e.g. 0.7)
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rays) return;

    float angle = yaw - fov_rad * 0.5f
                + fov_rad * (float)r / fmaxf((float)(n_rays - 1), 1.0f);
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
        if (cx < 0 || cx >= width || cy < 0 || cy >= height) break;

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

// Host wrapper for FOV grid update
inline void fov_grid_update(
    OccupancyGrid2D& grid,
    float2 uav_pos,
    float yaw,
    float fov_rad    = 1.57f,
    float max_range  = 2.5f,
    int   n_rays     = 64,
    float ray_step   = 0.1f,
    float free_update  = 0.01f,
    float occ_update   = 0.99f,
    float occ_threshold = 0.7f
) {
    dim3 block(256);
    dim3 grid_dim((n_rays + block.x - 1) / block.x);

    fov_grid_update_kernel<<<grid_dim, block>>>(
        grid.data, grid.dims.x, grid.dims.y,
        grid.origin, grid.resolution,
        uav_pos, yaw, fov_rad, max_range, n_rays, ray_step,
        free_update, occ_update, occ_threshold
    );
    cudaDeviceSynchronize();
}

} // namespace mppi

#endif // MPPI_FSMI_CUH
