#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mppi/controllers/mppi.cuh"
#include "mppi/controllers/smppi.cuh"
#include "mppi/controllers/kmppi.cuh"
#include "mppi/instantiations/double_integrator.cuh"
#include "mppi/instantiations/point_mass_3d.cuh"
#include "uav_control/mppi/acceleration_cost.cuh"

using namespace mppi;
using namespace mppi::instantiations;
using uav_control::mppi::AccelerationTrackingCost;

// RAII device ref trajectory (same as in mppi_gtest.cu)
struct DeviceRefTrajectory {
  float* d_ptr = nullptr;
  int horizon = 0;
  ~DeviceRefTrajectory() { if (d_ptr) cudaFree(d_ptr); }
  void upload(const std::vector<float>& data, int h) {
    horizon = h;
    if (d_ptr) cudaFree(d_ptr);
    HANDLE_ERROR(cudaMalloc(&d_ptr, data.size() * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_ptr, data.data(),
                            data.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  }
  void set_constant(const float ref[6], int h) {
    std::vector<float> flat(h * 6);
    for (int t = 0; t < h; ++t)
      for (int i = 0; i < 6; ++i)
        flat[t * 6 + i] = ref[i];
    upload(flat, h);
  }
};

static MPPIConfig make_acc_config() {
  MPPIConfig c{};
  c.num_samples     = 1024;
  c.horizon         = 50;
  c.nx              = PointMass3D::STATE_DIM;
  c.nu              = PointMass3D::CONTROL_DIM;
  c.lambda          = 1.0f;
  c.dt              = 0.02f;
  c.u_scale         = 1.0f;
  c.learning_rate   = 1.0f;
  c.w_action_seq_cost = 0.0f;
  c.num_support_pts = 10;
  c.control_sigma[0] = 1.0f;
  c.control_sigma[1] = 1.0f;
  c.control_sigma[2] = 1.0f;
  return c;
}

// ---------------------------------------------------------------------------
// 1. DoubleIntegrator convergence — MPPI vs SMPPI vs KMPPI
// ---------------------------------------------------------------------------
void log_di_convergence(const std::string& dir) {
  MPPIConfig config{};
  config.num_samples    = 512;
  config.horizon        = 30;
  config.nx             = 4;
  config.nu             = 2;
  config.lambda         = 1.0f;
  config.dt             = 0.05f;
  config.u_scale        = 1.0f;
  config.learning_rate  = 1.0f;
  config.w_action_seq_cost = 0.5f;
  config.num_support_pts   = 10;
  config.control_sigma[0]  = 1.0f;
  config.control_sigma[1]  = 1.0f;

  DoubleIntegrator dyn;
  QuadraticCost cost;

  struct Run { std::string name; };
  std::vector<Run> runs = {{"MPPI"}, {"SMPPI"}, {"KMPPI"}};

  for (auto& run : runs) {
    Eigen::VectorXf state(4);
    state << 2.0f, 2.0f, 0.0f, 0.0f;

    std::ofstream f(dir + "/di_" + run.name + ".csv");
    f << "t,px,py,vx,vy,ax,ay\n";

    // Create controller based on name
    MPPIController<DoubleIntegrator, QuadraticCost>* mppi = nullptr;
    SMPPIController<DoubleIntegrator, QuadraticCost>* smppi = nullptr;
    KMPPIController<DoubleIntegrator, QuadraticCost>* kmppi = nullptr;

    if (run.name == "MPPI")  mppi  = new MPPIController<DoubleIntegrator, QuadraticCost>(config, dyn, cost);
    if (run.name == "SMPPI") smppi = new SMPPIController<DoubleIntegrator, QuadraticCost>(config, dyn, cost);
    if (run.name == "KMPPI") kmppi = new KMPPIController<DoubleIntegrator, QuadraticCost>(config, dyn, cost);

    for (int t = 0; t < 80; ++t) {
      Eigen::VectorXf action(2);

      if (mppi)  { mppi->compute(state);  action = mppi->get_action();  mppi->set_applied_control(action);  mppi->shift(); }
      if (smppi) { smppi->compute(state); action = smppi->get_action(); smppi->set_applied_control(action); smppi->shift(); }
      if (kmppi) { kmppi->compute(state); action = kmppi->get_action(); kmppi->set_applied_control(action); }

      f << t * config.dt << "," << state(0) << "," << state(1) << ","
        << state(2) << "," << state(3) << "," << action(0) << "," << action(1) << "\n";

      float next[4];
      dyn.step(state.data(), action.data(), next, config.dt);
      state = Eigen::Map<Eigen::VectorXf>(next, 4);
    }

    delete mppi; delete smppi; delete kmppi;
    std::cout << "  Wrote " << run.name << std::endl;
  }
}

// ---------------------------------------------------------------------------
// 2. StaticPointTracking — AccelerationTrackingCost
// ---------------------------------------------------------------------------
void log_static_tracking(const std::string& dir) {
  auto config = make_acc_config();
  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 10.0f; cost.Q_vel = 1.0f; cost.R_acc = 0.01f;
  cost.R_du = 0.5f; cost.terminal_multiplier = 10.0f;

  DeviceRefTrajectory ref;
  float goal[6] = {3.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f};
  ref.set_constant(goal, config.horizon);
  cost.ref_traj = ref.d_ptr;
  cost.ref_horizon = ref.horizon;

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);
  Eigen::VectorXf state = Eigen::VectorXf::Zero(6);

  std::ofstream f(dir + "/static_tracking.csv");
  f << "t,px,py,pz,vx,vy,vz,ax,ay,az,ref_px,ref_py,ref_pz\n";

  for (int t = 0; t < 150; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();

    f << t * config.dt;
    for (int i = 0; i < 6; ++i) f << "," << state(i);
    for (int i = 0; i < 3; ++i) f << "," << action(i);
    f << "," << goal[0] << "," << goal[1] << "," << goal[2] << "\n";

    dyn.step_host(state, action, config.dt);
  }
  std::cout << "  Wrote static_tracking" << std::endl;
}

// ---------------------------------------------------------------------------
// 3. WaypointSequence — time-varying reference
// ---------------------------------------------------------------------------
void log_waypoint_sequence(const std::string& dir) {
  auto config = make_acc_config();
  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 10.0f; cost.Q_vel = 1.0f; cost.R_acc = 0.01f;
  cost.R_du = 0.5f; cost.terminal_multiplier = 10.0f;

  Eigen::Vector3f pos_A(0.0f, 0.0f, 0.0f);
  Eigen::Vector3f pos_B(4.0f, 3.0f, -2.0f);

  DeviceRefTrajectory ref;
  std::vector<float> ref_flat(config.horizon * 6);
  for (int t = 0; t < config.horizon; ++t) {
    float alpha = float(t) / (config.horizon - 1);
    Eigen::Vector3f pos = pos_A + alpha * (pos_B - pos_A);
    Eigen::Vector3f vel = (pos_B - pos_A) / (config.horizon * config.dt);
    for (int i = 0; i < 3; ++i) {
      ref_flat[t * 6 + i]     = pos(i);
      ref_flat[t * 6 + 3 + i] = vel(i);
    }
  }
  ref.upload(ref_flat, config.horizon);
  cost.ref_traj = ref.d_ptr;
  cost.ref_horizon = ref.horizon;

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);
  Eigen::VectorXf state = Eigen::VectorXf::Zero(6);

  std::ofstream f(dir + "/waypoint_sequence.csv");
  f << "t,px,py,pz,vx,vy,vz,ax,ay,az,ref_px,ref_py,ref_pz\n";

  for (int t = 0; t < 200; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();

    // Reference at this sim step (clamp to horizon)
    int ri = std::min(t, config.horizon - 1);
    f << t * config.dt;
    for (int i = 0; i < 6; ++i) f << "," << state(i);
    for (int i = 0; i < 3; ++i) f << "," << action(i);
    f << "," << ref_flat[ri*6+0] << "," << ref_flat[ri*6+1] << "," << ref_flat[ri*6+2] << "\n";

    dyn.step_host(state, action, config.dt);
  }
  std::cout << "  Wrote waypoint_sequence" << std::endl;
}

// ---------------------------------------------------------------------------
// 4. HoverAtOrigin — nullptr fallback
// ---------------------------------------------------------------------------
void log_hover(const std::string& dir) {
  auto config = make_acc_config();
  config.control_sigma[0] = 2.0f;
  config.control_sigma[1] = 2.0f;
  config.control_sigma[2] = 2.0f;
  config.num_samples = 2048;

  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 15.0f; cost.Q_vel = 2.0f; cost.R_acc = 0.01f;
  cost.R_du = 0.1f; cost.terminal_multiplier = 15.0f;

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);

  Eigen::VectorXf state(6);
  state << 2.0f, -1.0f, 0.5f, 0.0f, 0.0f, 0.0f;

  std::ofstream f(dir + "/hover_fallback.csv");
  f << "t,px,py,pz,vx,vy,vz,ax,ay,az\n";

  for (int t = 0; t < 200; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();

    f << t * config.dt;
    for (int i = 0; i < 6; ++i) f << "," << state(i);
    for (int i = 0; i < 3; ++i) f << "," << action(i);
    f << "\n";

    dyn.step_host(state, action, config.dt);
  }
  std::cout << "  Wrote hover_fallback" << std::endl;
}

int main() {
  std::string dir = DATA_DIR "/csv";
  system(("mkdir -p " + dir).c_str());

  std::cout << "Logging DoubleIntegrator convergence..." << std::endl;
  log_di_convergence(dir);

  std::cout << "Logging static point tracking..." << std::endl;
  log_static_tracking(dir);

  std::cout << "Logging waypoint sequence..." << std::endl;
  log_waypoint_sequence(dir);

  std::cout << "Logging hover fallback..." << std::endl;
  log_hover(dir);

  std::cout << "\nDone. CSVs in " << dir << "/" << std::endl;
  return 0;
}
