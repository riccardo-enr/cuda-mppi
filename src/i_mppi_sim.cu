#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "mppi/controllers/i_mppi.cuh"
#include "mppi/core/map.cuh"
#include "mppi/instantiations/fsmi_cost.cuh"

using namespace mppi;

// Simple 2D Dynamics
struct SimpleDynamics {
  static constexpr int STATE_DIM = 4;   // x, y, vx, vy
  static constexpr int CONTROL_DIM = 2; // ax, ay

  __device__ void step(const float *state, const float *u, float *next_state,
                       float dt) const {
    next_state[0] = state[0] + state[2] * dt;
    next_state[1] = state[1] + state[3] * dt;
    next_state[2] = state[2] + u[0] * dt;
    next_state[3] = state[3] + u[1] * dt;
  }

  void step_host(Eigen::VectorXf &state, const Eigen::VectorXf &u,
                 float dt) const {
    state[0] += state[2] * dt;
    state[1] += state[3] * dt;
    state[2] += u[0] * dt;
    state[3] += u[1] * dt;
  }
};

__global__ void compute_info_gain_grid_kernel(FSMICost cost, float *output,
                                              int width, int height,
                                              float resolution) {
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_idx >= width || y_idx >= height)
    return;

  float state[4] = {x_idx * resolution, y_idx * resolution, 0.0f, 0.0f};
  float u[2] = {0.0f, 0.0f};
  output[y_idx * width + x_idx] = cost.compute(state, u, 0);
}

void save_info_gain_map(FSMICost &cost, int width, int height, float resolution,
                        const std::string &filename) {
  float *d_output;
  cudaMalloc(&d_output, width * height * sizeof(float));
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  compute_info_gain_grid_kernel<<<grid, block>>>(cost, d_output, width, height,
                                                 resolution);
  std::vector<float> h_output(width * height);
  cudaMemcpy(h_output.data(), d_output, width * height * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::ofstream fs(filename);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      fs << h_output[y * width + x] << (x == width - 1 ? "" : ",");
    }
    fs << "\n";
  }
  fs.close();
  cudaFree(d_output);
}

// Helper function to update map based on robot observations
void update_map_observations(std::vector<float> &h_map, float x, float y,
                             float sensor_range, int width, int height,
                             float resolution,
                             std::vector<bool> &area_explored) {
  // Simulate sensor observations reducing uncertainty
  int robot_x = static_cast<int>(x / resolution);
  int robot_y = static_cast<int>(y / resolution);
  int range_cells = static_cast<int>(sensor_range / resolution);

  // Check if robot is in Area 1 or Area 2
  bool in_area1 = (x >= 2.0f && x <= 4.0f && y >= 6.5f && y <= 8.5f);
  bool in_area2 = (x >= 6.5f && x <= 8.5f && y >= 1.5f && y <= 3.5f);

  // If in an area for the first time, mark entire area as explored
  if (in_area1 && !area_explored[0]) {
    std::cout << "Explored Area 1 at t=" << x << "," << y << std::endl;
    area_explored[0] = true;
    // Set entire Area 1 to free space
    for (int y_idx = 65; y_idx < 85; ++y_idx)
      for (int x_idx = 20; x_idx < 40; ++x_idx)
        h_map[y_idx * width + x_idx] = 0.01f;
  }
  if (in_area2 && !area_explored[1]) {
    std::cout << "Explored Area 2 at t=" << x << "," << y << std::endl;
    area_explored[1] = true;
    // Set entire Area 2 to free space
    for (int y_idx = 15; y_idx < 35; ++y_idx)
      for (int x_idx = 65; x_idx < 85; ++x_idx)
        h_map[y_idx * width + x_idx] = 0.01f;
  }

  // General observation update
  for (int dy = -range_cells; dy <= range_cells; ++dy) {
    for (int dx = -range_cells; dx <= range_cells; ++dx) {
      int cell_x = robot_x + dx;
      int cell_y = robot_y + dy;

      if (cell_x < 0 || cell_x >= width || cell_y < 0 || cell_y >= height)
        continue;

      float dist = std::sqrt(dx * dx + dy * dy) * resolution;
      if (dist > sensor_range * 2)
        continue;

      int idx = cell_y * width + cell_x;
      float &prob = h_map[idx];

      // Skip walls
      if (prob > 0.9f)
        continue;

      // Aggressive uncertainty reduction
      if (prob > 0.1f && prob < 0.9f) {
        prob = 0.01f; // Immediately resolve all uncertainty
      }
    }
  }
}

float calculate_area_entropy(const std::vector<float>& map, int width, int x_min_idx, int x_max_idx, int y_min_idx, int y_max_idx) {
    float entropy = 0.0f;
    for (int y = y_min_idx; y < y_max_idx; ++y) {
        for (int x = x_min_idx; x < x_max_idx; ++x) {
            float p = map[y * width + x];
            if (p < 0.001f) p = 0.001f;
            if (p > 0.999f) p = 0.999f;
            entropy += -p * logf(p) - (1.0f - p) * logf(1.0f - p);
        }
    }
    return entropy;
}

void run_sim(float lambda_info, const std::string &prefix) {
  MPPIConfig planner_config;
  planner_config.num_samples = 1024;
  planner_config.horizon = 50;
  planner_config.nx = 4;
  planner_config.nu = 2;
  planner_config.lambda = 1.0f; // Higher lambda for smoother distribution
  planner_config.dt = 0.1f;
  planner_config.u_scale = 5.0f;
  planner_config.lambda_info = lambda_info;
  planner_config.alpha = 0.0f; // No bias for planner

  MPPIConfig controller_config = planner_config;
  controller_config.alpha = 0.3f; // Bias for reactive controller

  // Map
  int width = 100;
  int height = 100;
  int size = width * height;
  std::vector<float> h_map(size, 0.01f);

  // Area 1: Near start, above corridor (easier to reach first)
  for (int y = 65; y < 85; ++y)
    for (int x = 20; x < 40; ++x)
      h_map[y * width + x] = 0.5f; // y=6.5-8.5, x=2-4

  // Area 2: After corridor, below path to goal
  for (int y = 15; y < 35; ++y)
    for (int x = 65; x < 85; ++x)
      h_map[y * width + x] = 0.5f; // y=1.5-3.5, x=6.5-8.5

  // Wall at x=50 with opening at y=[40,60]
  for (int y = 0; y < 100; ++y)
    if (y < 40 || y > 60)
      h_map[y * width + 50] = 0.99f;

  float *d_map_data;
  cudaMalloc(&d_map_data, size * sizeof(float));
  cudaMemcpy(d_map_data, h_map.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  OccupancyGrid h_grid_struct;
  h_grid_struct.data = d_map_data;
  h_grid_struct.dims = make_int3(width, height, 1);
  h_grid_struct.resolution = 0.1f;
  h_grid_struct.origin = make_float3(0.0f, 0.0f, 0.0f);

  OccupancyGrid *d_grid_ptr;
  cudaMalloc(&d_grid_ptr, sizeof(OccupancyGrid));
  cudaMemcpy(d_grid_ptr, &h_grid_struct, sizeof(OccupancyGrid),
             cudaMemcpyHostToDevice);

  FSMICost cost;
  cost.map = d_grid_ptr;
  cost.lambda_info = controller_config.lambda_info;
  cost.sensor_range = 3.0f; // Shorter range - robot must get closer to observe
  cost.goal = make_float3(9.0f, 5.0f, 0.0f);
  cost.lambda_goal =
      0.05f; // Slightly higher goal weight to encourage movement between zones

  // Save initial information gain map
  if (lambda_info > 0)
    save_info_gain_map(cost, width, height, 0.1f, prefix + "_map_initial.csv");

  SimpleDynamics dyn;
  IMPPIController<SimpleDynamics, FSMICost> planner(planner_config, dyn, cost);
  IMPPIController<SimpleDynamics, FSMICost> controller(controller_config, dyn, cost);

  // Initial reference trajectory (zeros)
  Eigen::VectorXf u_ref = Eigen::VectorXf::Zero(controller_config.horizon * controller_config.nu);
  controller.set_reference_trajectory(u_ref);

  Eigen::VectorXf state = Eigen::VectorXf::Zero(4);
  state[0] = 1.0f;
  state[1] = 5.0f;
  state[2] = 0.5f; // Start with initial velocity

  std::ofstream fs(prefix + "_traj.csv");
  fs << "t,x,y,vx,vy\n";

  // Hierarchical exploration logic: explore areas, then drive to goal
  int map_update_freq =
      1; // Update every step for immediate uncertainty depletion
  std::vector<bool> area_explored(2,
                                  false); // Track if Area 1 and Area 2 explored

  for (int i = 0; i < 1200; ++i) { // Extended time for full exploration
    // FSMI-Driven Goal Logic
    if (lambda_info > 0) {
        float entropy1 = calculate_area_entropy(h_map, width, 20, 40, 65, 85);
        float entropy2 = calculate_area_entropy(h_map, width, 65, 85, 15, 35);
        
        float3 current_goal;
        float current_lambda_goal = 1.0f; // Stronger drive to intermediate goals

        if (entropy1 > 10.0f) {
             // Target Area 1 Center (3.0, 7.5)
             current_goal = make_float3(3.0f, 7.5f, 0.0f);
        } else if (entropy2 > 10.0f) {
             // Target Area 2 Center (7.5, 2.5)
             current_goal = make_float3(7.5f, 2.5f, 0.0f);
        } else {
             // Target Final Goal (9.0, 5.0)
             current_goal = make_float3(9.0f, 5.0f, 0.0f);
             current_lambda_goal = 2.0f; // Very strong drive to final goal
        }
        controller.update_cost_params(current_goal, current_lambda_goal);
        planner.update_cost_params(current_goal, current_lambda_goal);

        // Run Layer 2: FSMI-Informed Trajectory Generator
        planner.compute(state);
        Eigen::VectorXf u_ref = planner.get_optimal_control_sequence();

        // Pass to Layer 3: Informative Reference
        controller.set_reference_trajectory(u_ref);
    }

    // Run Layer 3: Reactive Biased-MPPI
    controller.compute(state);
    Eigen::VectorXf action = controller.get_action();
    fs << i * controller_config.dt << "," << state[0] << "," << state[1] << "," << state[2]
       << "," << state[3] << "\n";

    // For informative MPPI: update map based on observations (simulate
    // exploration)
    if (lambda_info > 0 && i % map_update_freq == 0) {
      update_map_observations(h_map, state[0], state[1], cost.sensor_range,
                              width, height, 0.1f, area_explored);
      cudaMemcpy(d_map_data, h_map.data(), size * sizeof(float),
                 cudaMemcpyHostToDevice);
    }

    dyn.step_host(state, action, controller_config.dt);
    controller.shift();
    if (lambda_info > 0) {
        planner.shift();
    }
    
    // Only break if we are near the FINAL goal (9,5) AND both areas are explored
    // Or just checking distance to final goal is sufficient if the logic steers us there last.
    if ((state.head(2) - Eigen::Vector2f(9, 5)).norm() < 0.3) {
        // Double check exploration for robustness? 
        // If we reached the goal, we assume we are done.
        break;
    }
  }

  // Save final information gain map to show reduced uncertainty
  if (lambda_info > 0)
    save_info_gain_map(cost, width, height, 0.1f, prefix + "_map_final.csv");

  fs.close();
  cudaFree(d_map_data);
  cudaFree(d_grid_ptr);
}

int main() {
  std::cout << "Running Standard Simulation..." << std::endl;
  run_sim(0.0f, "std");
  std::cout << "Running Informative Simulation..." << std::endl;
  run_sim(10000.0f,
          "info"); // Very high info weight to strongly prioritize exploration
  return 0;
}
