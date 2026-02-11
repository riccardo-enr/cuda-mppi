#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include "mppi/controllers/i_mppi.cuh"
#include "mppi/core/fsmi.cuh"
#include "mppi/instantiations/quadrotor.cuh"
#include "mppi/instantiations/informative_cost.cuh"
#include "mppi/planning/trajectory_generator.hpp"

using namespace mppi;
using namespace mppi::instantiations;
using namespace mppi::planning;

// ---------------------------------------------------------------------------
// Environment setup (matching Python env_setup.py)
// ---------------------------------------------------------------------------

// Create a 140×120 cell occupancy grid at 0.1 m/cell (14 m × 12 m)
static constexpr int GRID_W   = 140;
static constexpr int GRID_H   = 120;
static constexpr float GRID_RES = 0.1f;

void create_occupancy_grid(std::vector<float>& h_map) {
    h_map.assign(GRID_W * GRID_H, 0.0f);  // free space

    auto set_cell = [&](int cx, int cy, float val) {
        if (cx >= 0 && cx < GRID_W && cy >= 0 && cy < GRID_H)
            h_map[cy * GRID_W + cx] = val;
    };

    // Walls at occupancy 0.9
    auto fill_rect = [&](float x0, float y0, float x1, float y1, float val) {
        int ix0 = (int)(x0 / GRID_RES), iy0 = (int)(y0 / GRID_RES);
        int ix1 = (int)(x1 / GRID_RES), iy1 = (int)(y1 / GRID_RES);
        for (int y = iy0; y <= iy1; ++y)
            for (int x = ix0; x <= ix1; ++x)
                set_cell(x, y, val);
    };

    // Outer boundary walls
    fill_rect(0.0f, 2.0f, 4.0f, 2.2f, 0.9f);   // bottom wall first segment
    fill_rect(0.0f, 7.8f, 4.0f, 8.0f, 0.9f);    // top wall first segment
    fill_rect(3.8f, 0.0f, 4.0f, 2.0f, 0.9f);    // corner down
    fill_rect(3.8f, 8.0f, 4.0f, 10.0f, 0.9f);   // corner up
    fill_rect(4.0f, 0.0f, 12.0f, 0.2f, 0.9f);   // bottom long wall
    fill_rect(4.0f, 9.8f, 12.0f, 10.0f, 0.9f);  // top long wall
    fill_rect(11.8f, 0.0f, 12.0f, 10.0f, 0.9f); // end wall

    // Info zones at 0.5 (unknown)
    fill_rect(1.0f, 4.0f, 4.0f, 8.0f, 0.5f);    // bottom-left room
    fill_rect(10.0f, 4.0f, 12.0f, 8.0f, 0.5f);  // bottom-right room
    fill_rect(10.0f, 1.0f, 12.0f, 3.0f, 0.5f);  // top-right room
}

// ---------------------------------------------------------------------------
// Simulation constants
// ---------------------------------------------------------------------------
static constexpr float DT          = 0.05f;  // 50 Hz control
static constexpr int   SIM_STEPS   = 2000;   // 100 seconds
static constexpr int   FIELD_INTERVAL = 10;  // recompute field every 10 steps (5 Hz)

// ---------------------------------------------------------------------------
// Main simulation
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== I-MPPI Full Informative Simulation ===" << std::endl;

    // --- Grid setup ---
    std::vector<float> h_map;
    create_occupancy_grid(h_map);

    float* d_map;
    cudaMalloc(&d_map, GRID_W * GRID_H * sizeof(float));
    cudaMemcpy(d_map, h_map.data(), GRID_W * GRID_H * sizeof(float),
               cudaMemcpyHostToDevice);

    OccupancyGrid2D grid;
    grid.data       = d_map;
    grid.dims       = make_int2(GRID_W, GRID_H);
    grid.resolution = GRID_RES;
    grid.origin     = make_float2(0.0f, 0.0f);

    // --- FSMI & Field config ---
    FSMIConfig fsmi_cfg;
    fsmi_cfg.num_beams = 12;
    fsmi_cfg.max_range = 10.0f;
    fsmi_cfg.ray_step  = 0.1f;

    InfoFieldConfig ifc;
    ifc.field_res    = 0.5f;
    ifc.field_extent = 5.0f;
    ifc.n_yaw        = 8;
    ifc.lambda_info  = 20.0f;
    ifc.lambda_local = 10.0f;
    ifc.ref_speed    = 2.0f;
    ifc.ref_horizon  = 40;

    UniformFSMIConfig uniform_cfg;
    uniform_cfg.num_beams = 6;
    uniform_cfg.max_range = 2.5f;
    uniform_cfg.ray_step  = 0.2f;

    // --- Info field ---
    InfoField info_field;

    // --- Controller config ---
    MPPIConfig mppi_cfg;
    mppi_cfg.num_samples = 1024;
    mppi_cfg.horizon     = 40;
    mppi_cfg.nx          = 13;
    mppi_cfg.nu          = 4;
    mppi_cfg.lambda      = 0.1f;
    mppi_cfg.dt          = DT;
    mppi_cfg.u_scale     = 10.0f;  // noise * u_scale covers [0, 39.24] thrust range
    mppi_cfg.lambda_info = ifc.lambda_info;
    mppi_cfg.alpha       = 0.5f;

    // --- Dynamics & Cost ---
    QuadrotorDynamics dynamics;

    InformativeCost cost;
    cost.grid         = grid;
    cost.uniform_cfg  = uniform_cfg;
    cost.lambda_info  = ifc.lambda_info;
    cost.lambda_local = ifc.lambda_local;
    cost.target_weight = ifc.target_weight;
    cost.goal_weight   = ifc.goal_weight;
    cost.goal          = make_float3(9.0f, 5.0f, -2.0f);

    // --- Controller ---
    IMPPIController<QuadrotorDynamics, InformativeCost> controller(mppi_cfg, dynamics, cost);

    // Initialize nominal controls to hover: u_nom[t][0] = mg / u_scale
    {
        float hover_thrust_scaled = dynamics.mass * dynamics.gravity / mppi_cfg.u_scale;
        Eigen::VectorXf u_hover = Eigen::VectorXf::Zero(mppi_cfg.horizon * mppi_cfg.nu);
        for (int t = 0; t < mppi_cfg.horizon; ++t)
            u_hover(t * mppi_cfg.nu + 0) = hover_thrust_scaled;
        // Use set_reference_trajectory to set biased reference, and manually init u_nom
        controller.set_reference_trajectory(u_hover);
        // Also upload as initial u_nom via a temporary compute-free path
        cudaMemcpy(controller.get_u_nom_ptr(), u_hover.data(),
                   mppi_cfg.horizon * mppi_cfg.nu * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    // --- Trajectory generator ---
    TrajectoryGeneratorConfig tg_cfg;
    tg_cfg.ref_speed = ifc.ref_speed;

    std::vector<InfoZone> zones = {
        {2.5f, 6.0f, 3.0f, 4.0f, 100.0f},
        {11.5f, 6.0f, 3.0f, 4.0f, 100.0f},
        {11.5f, 2.0f, 3.0f, 2.0f, 100.0f},
    };

    TrajectoryGenerator traj_gen(tg_cfg, zones);
    std::vector<float> info_levels = {100.0f, 100.0f, 100.0f};

    // --- Initial state ---
    Eigen::VectorXf state = Eigen::VectorXf::Zero(13);
    state(0) = 1.0f;   // x
    state(1) = 5.0f;   // y
    state(2) = -2.0f;  // z (NED: up is negative)
    state(6) = 1.0f;   // qw (identity quaternion)

    // --- Reference trajectory (device) ---
    float* d_ref_traj;
    cudaMalloc(&d_ref_traj, ifc.ref_horizon * 3 * sizeof(float));

    // --- Logging ---
    std::ofstream traj_log("informative_sim_traj.csv");
    traj_log << "step,x,y,z,vx,vy,vz,yaw,info0,info1,info2\n";

    std::vector<float> h_field;  // for gradient trajectory computation

    // --- Simulation loop ---
    for (int step = 0; step < SIM_STEPS; ++step) {
        float px = state(0), py = state(1);

        // 1. FOV grid update
        float qw = state(6), qx = state(7), qy = state(8), qz = state(9);
        float yaw = atan2f(2.0f*(qw*qz + qx*qy), 1.0f - 2.0f*(qy*qy + qz*qz));

        fov_grid_update(grid, make_float2(px, py), yaw,
                        1.57f, 2.5f, 64, 0.1f, 0.01f, 0.99f, 0.7f);

        // 2. Every FIELD_INTERVAL steps: recompute info field + ref trajectory
        if (step % FIELD_INTERVAL == 0) {
            info_field.compute(grid, make_float2(px, py), ifc, fsmi_cfg);

            // Download field for host-side gradient trajectory
            h_field.resize(info_field.Nx * info_field.Ny);
            info_field.download(h_field.data());

            // Generate reference trajectory via gradient ascent
            std::vector<float> ref = traj_gen.field_gradient_trajectory(
                h_field.data(), info_field.Nx, info_field.Ny,
                info_field.origin.x, info_field.origin.y, info_field.res,
                px, py, ifc.ref_horizon, ifc.ref_speed, DT, -2.0f
            );

            // Upload to device
            cudaMemcpy(d_ref_traj, ref.data(), ifc.ref_horizon * 3 * sizeof(float),
                       cudaMemcpyHostToDevice);

            // Update cost fields and propagate to controller
            cost.ref_trajectory = d_ref_traj;
            cost.ref_horizon    = ifc.ref_horizon;
            cost.info_field     = info_field;
            controller.set_cost(cost);

            // Set reference control trajectory for biased sampling
            Eigen::VectorXf u_ref = Eigen::VectorXf::Zero(mppi_cfg.horizon * mppi_cfg.nu);
            controller.set_reference_trajectory(u_ref);
        }

        // 3. Update info zone levels (host-side)
        // Simplified: deplete based on distance to zone centre
        for (size_t z = 0; z < zones.size(); ++z) {
            float dx = px - zones[z].cx;
            float dy = py - zones[z].cy;
            float dist = sqrtf(dx*dx + dy*dy);
            float half_diag = sqrtf(zones[z].width*zones[z].width +
                                    zones[z].height*zones[z].height) * 0.5f;
            if (dist < half_diag) {
                // In or near zone: deplete proportionally
                float coverage = fmaxf(0.0f, 1.0f - dist / half_diag);
                info_levels[z] *= (1.0f - 0.02f * coverage);
            }
        }

        // 4. MPPI compute
        controller.compute(state);
        Eigen::VectorXf action = controller.get_action() * mppi_cfg.u_scale;

        // 5. Log
        traj_log << step << ","
                 << px << "," << py << "," << state(2) << ","
                 << state(3) << "," << state(4) << "," << state(5) << ","
                 << yaw << ","
                 << info_levels[0] << "," << info_levels[1] << "," << info_levels[2]
                 << "\n";

        // 6. Step true dynamics
        dynamics.step_host(state, action, DT);
        controller.shift();

        // Check goal reached
        float gx = state(0) - 9.0f, gy = state(1) - 5.0f;
        bool all_depleted = (info_levels[0] < 20.0f &&
                             info_levels[1] < 20.0f &&
                             info_levels[2] < 20.0f);
        if (all_depleted && sqrtf(gx*gx + gy*gy) < 0.5f) {
            std::cout << "Goal reached at step " << step << std::endl;
            break;
        }

        if (step % 100 == 0) {
            std::cout << "Step " << step
                      << " pos=(" << px << "," << py << "," << state(2) << ")"
                      << " info=(" << info_levels[0] << "," << info_levels[1]
                      << "," << info_levels[2] << ")" << std::endl;
        }
    }

    traj_log.close();

    // Sync grid back to host and save
    cudaMemcpy(h_map.data(), d_map, GRID_W * GRID_H * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::ofstream grid_log("informative_sim_grid_final.csv");
    for (int y = 0; y < GRID_H; ++y) {
        for (int x = 0; x < GRID_W; ++x) {
            grid_log << h_map[y * GRID_W + x];
            if (x < GRID_W - 1) grid_log << ",";
        }
        grid_log << "\n";
    }
    grid_log.close();

    std::cout << "Simulation complete. Output: informative_sim_traj.csv, "
              << "informative_sim_grid_final.csv" << std::endl;

    // Cleanup
    info_field.free();
    cudaFree(d_map);
    cudaFree(d_ref_traj);

    return 0;
}
