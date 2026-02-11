#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "mppi/controllers/i_mppi.cuh"
#include "mppi/instantiations/fsmi_cost.cuh"
#include "mppi/core/map.cuh"

using namespace mppi;

// Simple 2D Dynamics
struct SimpleDynamics
{
  static constexpr int STATE_DIM = 4;   // x, y, vx, vy
  static constexpr int CONTROL_DIM = 2;   // ax, ay

  __device__ void step(const float * state, const float * u, float * next_state, float dt) const
  {
    next_state[0] = state[0] + state[2] * dt;
    next_state[1] = state[1] + state[3] * dt;
    next_state[2] = state[2] + u[0] * dt;
    next_state[3] = state[3] + u[1] * dt;
  }
};

int main()
{
    // 1. Setup Config
  MPPIConfig config;
  config.num_samples = 256;
  config.horizon = 50;
  config.nx = 4;
  config.nu = 2;
  config.lambda = 0.5f;
  config.dt = 0.05f;
  config.u_scale = 5.0f;   // Max accel

    // I-MPPI params
  config.lambda_info = 10.0f;   // High info gain reward
  config.alpha = 0.2f;          // 20% samples biased

    // 2. Setup Map on Device
  int width = 100;
  int height = 100;
  int size = width * height;
  std::vector<float> h_map(size, 0.5f);    // Unknown space

    // Create a "known free" corridor in the middle
  for(int y = 40; y < 60; ++y) {
    for(int x = 0; x < 100; ++x) {
      h_map[y * width + x] = 0.0f;     // Free space (p=0) -> Entropy=0
    }
  }

    // Create a "high uncertainty" blob at (80, 50) - Target
    // Already 0.5 everywhere else.

  float * d_map_data;
  cudaMalloc(&d_map_data, size * sizeof(float));
  cudaMemcpy(d_map_data, h_map.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  OccupancyGrid h_grid_struct;
  h_grid_struct.data = d_map_data;
  h_grid_struct.dims = make_int3(width, height, 1);
  h_grid_struct.resolution = 0.1f;
  h_grid_struct.origin = make_float3(0.0f, 0.0f, 0.0f);

  OccupancyGrid * d_grid_ptr;
  cudaMalloc(&d_grid_ptr, sizeof(OccupancyGrid));
  cudaMemcpy(d_grid_ptr, &h_grid_struct, sizeof(OccupancyGrid), cudaMemcpyHostToDevice);

    // 3. Setup Cost
  FSMICost cost;
  cost.map = d_grid_ptr;
  cost.lambda_info = config.lambda_info;
  cost.sensor_range = 5.0f;   // 5 meters

  SimpleDynamics dyn;

    // 4. Initialize Controller
  std::cout << "Initializing I-MPPI..." << std::endl;
  IMPPIController<SimpleDynamics, FSMICost> controller(config, dyn, cost);

    // 5. Set Reference Trajectory (Bias)
    // Create a dummy reference trajectory (e.g., moving forward)
  Eigen::VectorXf u_ref = Eigen::VectorXf::Zero(config.horizon * config.nu);
  for(int t = 0; t < config.horizon; ++t) {
    u_ref[t * 2 + 0] = 1.0f;   // Forward accel
  }
  controller.set_reference_trajectory(u_ref);

    // 6. Run Control Loop
  Eigen::VectorXf state = Eigen::VectorXf::Zero(4);
  state[0] = 1.0f;   // x
  state[1] = 5.0f;   // y (middle of corridor is 5.0 in world coords? 50 * 0.1 = 5.0)

  std::cout << "Running Compute..." << std::endl;
  controller.compute(state);

  Eigen::VectorXf action = controller.get_action();
  std::cout << "I-MPPI Action: " << action.transpose() << std::endl;

    // Cleanup
  cudaFree(d_map_data);
  cudaFree(d_grid_ptr);

  std::cout << "Done." << std::endl;
  return 0;
}
