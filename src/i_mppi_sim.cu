#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "mppi/controllers/i_mppi.cuh"
#include "mppi/instantiations/fsmi_cost.cuh"
#include "mppi/core/map.cuh"

using namespace mppi;

// Simple 2D Dynamics
struct SimpleDynamics {
    static constexpr int STATE_DIM = 4; // x, y, vx, vy
    static constexpr int CONTROL_DIM = 2; // ax, ay

    __device__ void step(const float* state, const float* u, float* next_state, float dt) const {
        next_state[0] = state[0] + state[2] * dt;
        next_state[1] = state[1] + state[3] * dt;
        next_state[2] = state[2] + u[0] * dt;
        next_state[3] = state[3] + u[1] * dt;
    }
    
    void step_host(Eigen::VectorXf& state, const Eigen::VectorXf& u, float dt) const {
        state[0] += state[2] * dt;
        state[1] += state[3] * dt;
        state[2] += u[0] * dt;
        state[3] += u[1] * dt;
    }
};

__global__ void compute_info_gain_grid_kernel(
    FSMICost cost,
    float* output,
    int width,
    int height,
    float resolution
) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x_idx >= width || y_idx >= height) return;
    
    float state[4] = {x_idx * resolution, y_idx * resolution, 0.0f, 0.0f};
    float u[2] = {0.0f, 0.0f};
    
    // We only care about the info gain part of the cost
    // We can call cost.compute_info_gain(state) if we expose it, 
    // or just cost.compute and ignore the rest if cost is simple.
    
    // For now, let's assume we can call compute. 
    // In FSMICost, compute returns lambda_info * info_gain + stage_cost
    output[y_idx * width + x_idx] = cost.compute(state, u, 0);
}

void save_info_gain_map(FSMICost& cost, int width, int height, float resolution, const std::string& filename) {
    float* d_output;
    cudaMalloc(&d_output, width * height * sizeof(float));
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    compute_info_gain_grid_kernel<<<grid, block>>>(cost, d_output, width, height, resolution);
    
    std::vector<float> h_output(width * height);
    cudaMemcpy(h_output.data(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::ofstream fs(filename);
    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            fs << h_output[y*width + x] << (x == width - 1 ? "" : ",");
        }
        fs << "\n";
    }
    fs.close();
    cudaFree(d_output);
}

void run_sim(float lambda_info, const std::string& prefix) {
    MPPIConfig config;
    config.num_samples = 512;
    config.horizon = 30;
    config.nx = 4;
    config.nu = 2;
    config.lambda = 0.1f;
    config.dt = 0.1f;
    config.u_scale = 10.0f;
    config.lambda_info = lambda_info;
    config.alpha = 0.2f;

    int width = 100;
    int height = 100;
    int size = width * height;
    std::vector<float> h_map(size, 0.5f);
    
    // Corridor from (0, 5) to (10, 5)
    for(int y=40; y<60; ++y) {
        for(int x=0; x<100; ++x) {
            h_map[y*width + x] = 0.01f; 
        }
    }
    // Add an obstacle/wall at x=50, with a small opening
    for(int y=0; y<100; ++y) {
        if (y < 45 || y > 55) {
            h_map[y*width + 50] = 0.99f; 
        }
    }
    
    float* d_map_data;
    cudaMalloc(&d_map_data, size * sizeof(float));
    cudaMemcpy(d_map_data, h_map.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    OccupancyGrid h_grid_struct;
    h_grid_struct.data = d_map_data;
    h_grid_struct.dims = make_int3(width, height, 1);
    h_grid_struct.resolution = 0.1f;
    h_grid_struct.origin = make_float3(0.0f, 0.0f, 0.0f);
    
    OccupancyGrid* d_grid_ptr;
    cudaMalloc(&d_grid_ptr, sizeof(OccupancyGrid));
    cudaMemcpy(d_grid_ptr, &h_grid_struct, sizeof(OccupancyGrid), cudaMemcpyHostToDevice);

    FSMICost cost;
    cost.map = d_grid_ptr;
    cost.lambda_info = config.lambda_info;
    cost.sensor_range = 4.0f;
    
    if (lambda_info > 0) {
        save_info_gain_map(cost, width, height, 0.1f, prefix + "_map.csv");
    }

    SimpleDynamics dyn;
    IMPPIController<SimpleDynamics, FSMICost> controller(config, dyn, cost);

    Eigen::VectorXf u_ref = Eigen::VectorXf::Zero(config.horizon * config.nu);
    for(int t=0; t<config.horizon; ++t) u_ref[t*2+0] = 0.1f; 
    controller.set_reference_trajectory(u_ref);

    Eigen::VectorXf state = Eigen::VectorXf::Zero(4);
    state[0] = 1.0f; 
    state[1] = 5.0f; 
    state[2] = 0.5f; 

    std::ofstream fs(prefix + "_traj.csv");
    fs << "t,x,y,vx,vy\n";

    for(int i=0; i<100; ++i) {
        controller.compute(state);
        Eigen::VectorXf action = controller.get_action();
        fs << i*config.dt << "," << state[0] << "," << state[1] << "," << state[2] << "," << state[3] << "\n";
        dyn.step_host(state, action, config.dt);
    }

    fs.close();
    cudaFree(d_map_data);
    cudaFree(d_grid_ptr);
}

int main() {
    std::cout << "Running Standard Simulation..." << std::endl;
    run_sim(0.0f, "std");
    
    std::cout << "Running Informative Simulation..." << std::endl;
    run_sim(50.0f, "info");
    
    return 0;
}
