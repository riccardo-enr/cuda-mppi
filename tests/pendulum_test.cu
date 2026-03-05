#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include "mppi/controllers/mppi.cuh"
#include "mppi/controllers/smppi.cuh"
#include "mppi/controllers/kmppi.cuh"
#include "mppi/instantiations/inverted_pendulum.cuh"

using namespace mppi;
using namespace mppi::instantiations;

void run_simulation(const std::string & controller_name)
{
  std::cout << "\nRunning " << controller_name << " Simulation..." << std::endl;

  MPPIConfig config;
  config.num_samples = 1024;
  config.horizon = 50;
  config.nx = InvertedPendulum::STATE_DIM;
  config.nu = InvertedPendulum::CONTROL_DIM;
  config.lambda = 0.5f;
  config.dt = 0.02f;
  config.u_scale = 1.0f;
  config.w_action_seq_cost = 0.0f;
  config.num_support_pts = 10;   // For KMPPI/SMPPI

  InvertedPendulum dyn;
  PendulumCost cost;

    // Initial state: perturbed from upright
    // [x, theta, x_dot, theta_dot]
  Eigen::VectorXf state(4);
  state << 0.0f, 0.1f, 0.0f, 0.0f;   // 0.1 rad (~5.7 deg) lean

    // Controller
    // We need a base class or template handling. Since the classes are templates, we can't easily swap them at runtime without polymorphism or template functions.
    // I'll just use if/else for the example or templates.
    // Since I want to print separate runs, I'll just instantiate inside specific blocks or templated function.
}

template<template<typename, typename> class ControllerT>
void run_controller(const std::string & name)
{
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Testing " << name << std::endl;

  MPPIConfig config;
  config.num_samples = 2048;
  config.horizon = 30;
  config.nx = InvertedPendulum::STATE_DIM;
  config.nu = InvertedPendulum::CONTROL_DIM;
  config.lambda = 10.0f;   // Higher lambda for sharper distribution
  config.dt = 0.02f;
  config.u_scale = 1.0f;
  config.w_action_seq_cost = 0.0f;   // No smoothing for now
  config.num_support_pts = 15;

  InvertedPendulum dyn;
  PendulumCost cost;

  ControllerT<InvertedPendulum, PendulumCost> controller(config, dyn, cost);

  Eigen::VectorXf state(4);
  state << 0.0f, 0.2f, 0.0f, 0.0f;   // Initial lean

  std::cout << "Initial State: " << state.transpose() << std::endl;

    // Sim loop
  float sim_time = 2.0f;
  int steps = (int)(sim_time / config.dt);

  for (int t = 0; t < steps; ++t) {
    controller.compute(state);
    Eigen::VectorXf action = controller.get_action();

        // Apply action to real dynamics (same as model here)
    float next_state_arr[4];
    dyn.step(state.data(), action.data(), next_state_arr, config.dt);

    state = Eigen::Map<Eigen::VectorXf>(next_state_arr, 4);

    if (t % 10 == 0) {
      std::cout << "t=" << t * config.dt << "s | x=" << state[0] << ", th=" << state[1] <<
        " | u=" << action[0] << std::endl;
    }
  }
  std::cout << "Final State: " << state.transpose() << std::endl;
}

int main()
{
  run_controller<MPPIController>("MPPI");
    // SMPPI and KMPPI might need tuning or different config, but let's try same config.
  run_controller<SMPPIController>("SMPPI");
  run_controller<KMPPIController>("KMPPI");

  return 0;
}
