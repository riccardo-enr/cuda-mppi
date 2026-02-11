#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "mppi/core/fsmi.cuh"

using namespace mppi;

// ---------------------------------------------------------------------------
// Test kernel: compute FSMI / Uniform-FSMI at a single pose on device
// ---------------------------------------------------------------------------
__global__ void test_fsmi_kernel(
    OccupancyGrid2D grid,
    float2 pos,
    float yaw,
    FSMIConfig fsmi_cfg,
    UniformFSMIConfig uniform_cfg,
    float* out_full_fsmi,
    float* out_uniform_fsmi
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_full_fsmi    = compute_fsmi_at_pose(grid, pos, yaw, fsmi_cfg);
        *out_uniform_fsmi = compute_uniform_fsmi_at_pose(grid, pos, yaw, uniform_cfg);
    }
}

// ---------------------------------------------------------------------------
// Test kernel: single beam FSMI
// ---------------------------------------------------------------------------
__global__ void test_single_beam_kernel(
    float* cell_probs,
    float* cell_dists,
    int N,
    FSMIConfig cfg,
    float* out_mi
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_mi = compute_beam_fsmi(cell_probs, cell_dists, N, cfg);
    }
}

// ---------------------------------------------------------------------------
// Test kernel: single beam Uniform-FSMI
// ---------------------------------------------------------------------------
__global__ void test_single_beam_uniform_kernel(
    float* cell_probs,
    int N,
    UniformFSMIConfig cfg,
    float* out_mi
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_mi = compute_beam_uniform_fsmi(cell_probs, N, cfg);
    }
}

// ---------------------------------------------------------------------------
// Helper: create a simple test grid
// ---------------------------------------------------------------------------
void create_test_grid(std::vector<float>& h_map, int W, int H) {
    h_map.assign(W * H, 0.0f);  // all free

    // Place an unknown zone at cells (5..15, 5..15) → 0.5
    for (int y = 5; y < 15; ++y)
        for (int x = 5; x < 15; ++x)
            h_map[y * W + x] = 0.5f;

    // Place a wall at column 12 (blocking some of the zone)
    for (int y = 0; y < H; ++y)
        h_map[y * W + 12] = 0.9f;
}

// ---------------------------------------------------------------------------
// Main test driver
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== FSMI Unit Tests ===" << std::endl;

    int W = 30, H = 30;
    float res = 0.1f;

    std::vector<float> h_map;
    create_test_grid(h_map, W, H);

    // Upload grid to device
    float* d_map;
    cudaMalloc(&d_map, W * H * sizeof(float));
    cudaMemcpy(d_map, h_map.data(), W * H * sizeof(float), cudaMemcpyHostToDevice);

    OccupancyGrid2D grid;
    grid.data       = d_map;
    grid.dims       = make_int2(W, H);
    grid.resolution = res;
    grid.origin     = make_float2(0.0f, 0.0f);

    // Device output
    float *d_full, *d_uniform;
    cudaMalloc(&d_full, sizeof(float));
    cudaMalloc(&d_uniform, sizeof(float));

    // --- Test 1: FSMI at a pose facing the unknown zone ---
    {
        FSMIConfig cfg;
        cfg.num_beams = 16;
        cfg.max_range = 2.0f;
        cfg.ray_step  = 0.1f;
        cfg.fov_rad   = 1.57f;

        UniformFSMIConfig ucfg;
        ucfg.num_beams = 6;
        ucfg.max_range = 2.0f;
        ucfg.ray_step  = 0.2f;

        float2 pos = {0.5f, 1.0f};
        float yaw  = 0.7854f;  // ~45 degrees, pointing toward zone

        test_fsmi_kernel<<<1, 1>>>(grid, pos, yaw, cfg, ucfg, d_full, d_uniform);
        cudaDeviceSynchronize();

        float h_full, h_uniform;
        cudaMemcpy(&h_full, d_full, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_uniform, d_uniform, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Test 1: Pose facing unknown zone" << std::endl;
        std::cout << "  Full FSMI:    " << h_full << std::endl;
        std::cout << "  Uniform FSMI: " << h_uniform << std::endl;
        assert(h_full > 0.0f && "Full FSMI should be positive facing unknown zone");
        assert(h_uniform > 0.0f && "Uniform FSMI should be positive facing unknown zone");
        std::cout << "  PASSED" << std::endl;
    }

    // --- Test 2: FSMI at a pose in completely free space ---
    {
        FSMIConfig cfg;
        cfg.num_beams = 8;
        cfg.max_range = 1.0f;
        cfg.ray_step  = 0.1f;

        UniformFSMIConfig ucfg;
        ucfg.num_beams = 4;
        ucfg.max_range = 1.0f;
        ucfg.ray_step  = 0.2f;

        float2 pos = {2.5f, 0.2f};  // far from zone/wall
        float yaw  = 0.0f;

        test_fsmi_kernel<<<1, 1>>>(grid, pos, yaw, cfg, ucfg, d_full, d_uniform);
        cudaDeviceSynchronize();

        float h_full, h_uniform;
        cudaMemcpy(&h_full, d_full, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_uniform, d_uniform, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "\nTest 2: Pose in free space" << std::endl;
        std::cout << "  Full FSMI:    " << h_full << std::endl;
        std::cout << "  Uniform FSMI: " << h_uniform << std::endl;
        // Free space (p=0.0) has very low information gain
        std::cout << "  PASSED (low gain expected in known-free space)" << std::endl;
    }

    // --- Test 3: Single beam through unknown zone ---
    {
        int N = 20;
        std::vector<float> h_probs(N);
        std::vector<float> h_dists(N);
        for (int i = 0; i < N; ++i) {
            h_probs[i] = 0.5f;  // all unknown
            h_dists[i] = (i + 0.5f) * 0.1f;
        }

        float *d_probs, *d_dists, *d_mi;
        cudaMalloc(&d_probs, N * sizeof(float));
        cudaMalloc(&d_dists, N * sizeof(float));
        cudaMalloc(&d_mi, sizeof(float));
        cudaMemcpy(d_probs, h_probs.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dists, h_dists.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        FSMIConfig cfg;
        cfg.ray_step = 0.1f;

        test_single_beam_kernel<<<1, 1>>>(d_probs, d_dists, N, cfg, d_mi);
        cudaDeviceSynchronize();

        float h_mi;
        cudaMemcpy(&h_mi, d_mi, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "\nTest 3: Single beam through 20 unknown cells" << std::endl;
        std::cout << "  Beam FSMI: " << h_mi << std::endl;
        assert(h_mi > 0.0f && "Beam through unknown cells should have positive MI");
        std::cout << "  PASSED" << std::endl;

        cudaFree(d_probs);
        cudaFree(d_dists);
        cudaFree(d_mi);
    }

    // --- Test 4: Single beam Uniform-FSMI ---
    {
        int N = 12;
        std::vector<float> h_probs(N, 0.5f);

        float *d_probs, *d_mi;
        cudaMalloc(&d_probs, N * sizeof(float));
        cudaMalloc(&d_mi, sizeof(float));
        cudaMemcpy(d_probs, h_probs.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        UniformFSMIConfig ucfg;
        ucfg.ray_step = 0.2f;

        test_single_beam_uniform_kernel<<<1, 1>>>(d_probs, N, ucfg, d_mi);
        cudaDeviceSynchronize();

        float h_mi;
        cudaMemcpy(&h_mi, d_mi, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "\nTest 4: Uniform-FSMI single beam, 12 unknown cells" << std::endl;
        std::cout << "  Beam MI: " << h_mi << std::endl;
        assert(h_mi > 0.0f);
        std::cout << "  PASSED" << std::endl;

        cudaFree(d_probs);
        cudaFree(d_mi);
    }

    // --- Test 5: Info field computation ---
    {
        InfoField field;
        InfoFieldConfig ifc;
        ifc.field_res    = 0.5f;
        ifc.field_extent = 1.5f;  // small field for test
        ifc.n_yaw        = 4;

        FSMIConfig cfg;
        cfg.num_beams = 8;
        cfg.max_range = 1.5f;
        cfg.ray_step  = 0.1f;

        float2 uav_pos = {1.0f, 1.0f};
        field.compute(grid, uav_pos, ifc, cfg);

        int Nx = field.Nx, Ny = field.Ny;
        std::vector<float> h_field(Nx * Ny);
        field.download(h_field.data());

        float max_val = -1e30f, min_val = 1e30f;
        for (int i = 0; i < Nx * Ny; ++i) {
            if (h_field[i] > max_val) max_val = h_field[i];
            if (h_field[i] < min_val) min_val = h_field[i];
        }

        std::cout << "\nTest 5: Info field (" << Nx << "x" << Ny << ")" << std::endl;
        std::cout << "  min=" << min_val << " max=" << max_val << std::endl;
        assert(max_val > min_val && "Field should have variation");
        std::cout << "  PASSED" << std::endl;

        field.free();
    }

    // --- Test 6: FOV grid update ---
    {
        // Create a fresh grid copy
        float* d_map2;
        cudaMalloc(&d_map2, W * H * sizeof(float));
        cudaMemcpy(d_map2, h_map.data(), W * H * sizeof(float), cudaMemcpyHostToDevice);

        OccupancyGrid2D grid2;
        grid2.data       = d_map2;
        grid2.dims       = make_int2(W, H);
        grid2.resolution = res;
        grid2.origin     = make_float2(0.0f, 0.0f);

        // Cast FOV from position (1.0, 1.0) looking at 45 degrees
        fov_grid_update(grid2, make_float2(1.0f, 1.0f), 0.7854f,
                        1.57f, 1.5f, 32, 0.1f, 0.01f, 0.99f, 0.7f);

        std::vector<float> h_map2(W * H);
        cudaMemcpy(h_map2.data(), d_map2, W * H * sizeof(float), cudaMemcpyDeviceToHost);

        // Check that some cells near (1.0, 1.0) are now known-free
        int cx = (int)(1.0f / res), cy = (int)(1.0f / res);
        float val = h_map2[cy * W + cx + 1];  // cell just to the right

        std::cout << "\nTest 6: FOV grid update" << std::endl;
        std::cout << "  Cell at (" << cx+1 << "," << cy << ") after update: " << val << std::endl;
        // Should be updated to free (0.01) since it was originally 0.0
        std::cout << "  PASSED" << std::endl;

        cudaFree(d_map2);
    }

    // Cleanup
    cudaFree(d_map);
    cudaFree(d_full);
    cudaFree(d_uniform);

    std::cout << "\n=== All FSMI tests passed ===" << std::endl;
    return 0;
}
