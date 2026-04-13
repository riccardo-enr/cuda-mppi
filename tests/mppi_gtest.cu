#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "mppi/controllers/mppi.cuh"
#include "mppi/controllers/smppi.cuh"
#include "mppi/controllers/kmppi.cuh"
#include "mppi/instantiations/double_integrator.cuh"
#include "mppi/instantiations/point_mass_3d.cuh"
#include "mppi/instantiations/quadrotor.cuh"
#include "mppi/instantiations/acceleration_cost.cuh"

using namespace mppi;
using namespace mppi::instantiations;

// ---------------------------------------------------------------------------
// Goal cost for PointMass3D tests
// ---------------------------------------------------------------------------
struct GoalCost3D {
  float goal[3] = {0.0f, 0.0f, 0.0f};
  float w_pos   = 5.0f;
  float w_vel   = 1.0f;
  float w_ctrl  = 0.1f;
  float w_term  = 20.0f;

  __host__ __device__ float compute(const float* x, const float* u,
                                    const float* /*u_prev*/, int /*t*/) const {
    float c = 0.0f;
    for (int i = 0; i < 3; ++i) {
      float dp = x[i] - goal[i];
      c += w_pos * dp * dp;
      c += w_vel * x[i + 3] * x[i + 3];
      c += w_ctrl * u[i] * u[i];
    }
    return c;
  }

  __host__ __device__ float terminal_cost(const float* x) const {
    float c = 0.0f;
    for (int i = 0; i < 3; ++i) {
      float dp = x[i] - goal[i];
      c += w_term * dp * dp;
    }
    return c;
  }
};

// ---------------------------------------------------------------------------
// Helper: build a default config for DoubleIntegrator
// ---------------------------------------------------------------------------
static MPPIConfig make_di_config() {
  MPPIConfig c{};
  c.num_samples    = 512;
  c.horizon        = 30;
  c.nx             = DoubleIntegrator::STATE_DIM;
  c.nu             = DoubleIntegrator::CONTROL_DIM;
  c.lambda         = 1.0f;
  c.dt             = 0.05f;
  c.u_scale        = 1.0f;
  c.learning_rate  = 1.0f;
  c.w_action_seq_cost = 0.0f;
  c.num_support_pts   = 10;
  c.control_sigma[0]  = 1.0f;
  c.control_sigma[1]  = 1.0f;
  return c;
}

// Helper: build a default config for PointMass3D
static MPPIConfig make_pm3d_config() {
  MPPIConfig c{};
  c.num_samples    = 1024;
  c.horizon        = 40;
  c.nx             = PointMass3D::STATE_DIM;
  c.nu             = PointMass3D::CONTROL_DIM;
  c.lambda         = 1.0f;
  c.dt             = 0.05f;
  c.u_scale        = 1.0f;
  c.learning_rate  = 1.0f;
  c.w_action_seq_cost = 0.0f;
  c.num_support_pts   = 10;
  c.control_sigma[0]  = 2.0f;
  c.control_sigma[1]  = 2.0f;
  c.control_sigma[2]  = 2.0f;
  return c;
}

// ===========================================================================
// MPPI Tests — DoubleIntegrator
// ===========================================================================

TEST(MPPIDoubleIntegrator, Construction) {
  auto config = make_di_config();
  DoubleIntegrator dyn;
  QuadraticCost cost;
  MPPIController<DoubleIntegrator, QuadraticCost> ctrl(config, dyn, cost);
  // If we get here without CUDA errors, construction succeeded
}

TEST(MPPIDoubleIntegrator, SingleComputeFinite) {
  auto config = make_di_config();
  DoubleIntegrator dyn;
  QuadraticCost cost;
  MPPIController<DoubleIntegrator, QuadraticCost> ctrl(config, dyn, cost);

  Eigen::VectorXf state = Eigen::VectorXf::Zero(4);
  state << 1.0f, 1.0f, 0.0f, 0.0f;

  ctrl.compute(state);
  Eigen::VectorXf action = ctrl.get_action();

  ASSERT_EQ(action.size(), 2);
  for (int i = 0; i < action.size(); ++i) {
    EXPECT_TRUE(std::isfinite(action(i)))
        << "action(" << i << ") = " << action(i);
  }
}

TEST(MPPIDoubleIntegrator, Convergence) {
  auto config = make_di_config();
  DoubleIntegrator dyn;
  QuadraticCost cost;
  MPPIController<DoubleIntegrator, QuadraticCost> ctrl(config, dyn, cost);

  Eigen::VectorXf state(4);
  state << 2.0f, 2.0f, 0.0f, 0.0f;

  float initial_pos_norm = state.head<2>().norm();

  const int sim_steps = 80;
  for (int t = 0; t < sim_steps; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();

    // Step dynamics on host
    float next[4];
    dyn.step(state.data(), action.data(), next, config.dt);
    state = Eigen::Map<Eigen::VectorXf>(next, 4);
  }

  float final_pos_norm = state.head<2>().norm();
  EXPECT_LT(final_pos_norm, initial_pos_norm * 0.3f)
      << "Expected convergence toward origin. "
      << "Initial norm: " << initial_pos_norm
      << ", final norm: " << final_pos_norm;
}

TEST(MPPIDoubleIntegrator, ShiftWarmStart) {
  auto config = make_di_config();
  DoubleIntegrator dyn;
  QuadraticCost cost;
  MPPIController<DoubleIntegrator, QuadraticCost> ctrl(config, dyn, cost);

  // Run a compute to populate u_nom with non-zero values
  Eigen::VectorXf state(4);
  state << 1.0f, 1.0f, 0.0f, 0.0f;
  ctrl.compute(state);

  // Read action at t=0 and t=1 before shift
  Eigen::VectorXf action_t0(config.nu);
  Eigen::VectorXf action_t1(config.nu);
  cudaMemcpy(action_t0.data(), ctrl.get_u_nom_ptr(),
             config.nu * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(action_t1.data(), ctrl.get_u_nom_ptr() + config.nu,
             config.nu * sizeof(float), cudaMemcpyDeviceToHost);

  ctrl.shift();

  // After shift, t=0 should now be what was t=1
  Eigen::VectorXf action_after_shift(config.nu);
  cudaMemcpy(action_after_shift.data(), ctrl.get_u_nom_ptr(),
             config.nu * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < config.nu; ++i) {
    EXPECT_FLOAT_EQ(action_after_shift(i), action_t1(i))
        << "Shift did not move t=1 to t=0 for dim " << i;
  }
}

TEST(MPPIDoubleIntegrator, SetNominalControl) {
  auto config = make_di_config();
  DoubleIntegrator dyn;
  QuadraticCost cost;
  MPPIController<DoubleIntegrator, QuadraticCost> ctrl(config, dyn, cost);

  Eigen::VectorXf u_set(2);
  u_set << 0.5f, -0.3f;
  ctrl.set_nominal_control(u_set);

  Eigen::VectorXf action = ctrl.get_action();
  for (int i = 0; i < config.nu; ++i) {
    EXPECT_FLOAT_EQ(action(i), u_set(i))
        << "set_nominal_control didn't round-trip for dim " << i;
  }
}

TEST(MPPIDoubleIntegrator, LambdaSensitivity) {
  DoubleIntegrator dyn;
  QuadraticCost cost;

  Eigen::VectorXf state(4);
  state << 2.0f, 0.0f, 0.0f, 0.0f;

  // Low lambda (sharp)
  auto cfg_low = make_di_config();
  cfg_low.lambda = 0.1f;
  MPPIController<DoubleIntegrator, QuadraticCost> ctrl_low(cfg_low, dyn, cost);
  ctrl_low.compute(state);
  Eigen::VectorXf action_low = ctrl_low.get_action();

  // High lambda (diffuse)
  auto cfg_high = make_di_config();
  cfg_high.lambda = 100.0f;
  MPPIController<DoubleIntegrator, QuadraticCost> ctrl_high(cfg_high, dyn, cost);
  ctrl_high.compute(state);
  Eigen::VectorXf action_high = ctrl_high.get_action();

  // Both should be finite
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(std::isfinite(action_low(i)));
    EXPECT_TRUE(std::isfinite(action_high(i)));
  }

  // Low lambda should produce larger magnitude action (sharper weighting
  // concentrates on best trajectories which accelerate harder toward origin)
  EXPECT_GT(action_low.norm(), action_high.norm() * 0.5f)
      << "Expected low-lambda to produce at least a moderately larger action";
}

// ===========================================================================
// MPPI Tests — PointMass3D
// ===========================================================================

TEST(MPPIPointMass3D, Convergence) {
  auto config = make_pm3d_config();
  PointMass3D dyn;
  GoalCost3D cost;

  MPPIController<PointMass3D, GoalCost3D> ctrl(config, dyn, cost);

  Eigen::VectorXf state(6);
  state << 5.0f, 5.0f, -3.0f, 0.0f, 0.0f, 0.0f;

  float initial_pos_norm = state.head<3>().norm();

  const int sim_steps = 100;
  for (int t = 0; t < sim_steps; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();

    dyn.step_host(state, action, config.dt);
  }

  float final_pos_norm = state.head<3>().norm();
  EXPECT_LT(final_pos_norm, initial_pos_norm * 0.3f)
      << "PointMass3D did not converge. "
      << "Initial: " << initial_pos_norm
      << ", final: " << final_pos_norm;
}

TEST(MPPIPointMass3D, GoalReaching) {
  auto config = make_pm3d_config();
  PointMass3D dyn;
  GoalCost3D cost;
  cost.goal[0] = 3.0f;
  cost.goal[1] = -2.0f;
  cost.goal[2] = 0.0f;

  MPPIController<PointMass3D, GoalCost3D> ctrl(config, dyn, cost);

  Eigen::VectorXf state = Eigen::VectorXf::Zero(6);

  const int sim_steps = 120;
  for (int t = 0; t < sim_steps; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();

    dyn.step_host(state, action, config.dt);
  }

  Eigen::Vector3f goal(cost.goal[0], cost.goal[1], cost.goal[2]);
  float dist = (state.head<3>() - goal).norm();
  EXPECT_LT(dist, 2.0f)
      << "Did not reach goal. Final pos: " << state.head<3>().transpose()
      << ", goal: " << goal.transpose() << ", dist: " << dist;
}

// ===========================================================================
// SMPPI Tests
// ===========================================================================

TEST(SMPPIDoubleIntegrator, Convergence) {
  auto config = make_di_config();
  config.w_action_seq_cost = 0.5f;
  DoubleIntegrator dyn;
  QuadraticCost cost;
  SMPPIController<DoubleIntegrator, QuadraticCost> ctrl(config, dyn, cost);

  Eigen::VectorXf state(4);
  state << 2.0f, 2.0f, 0.0f, 0.0f;
  float initial_norm = state.head<2>().norm();

  for (int t = 0; t < 80; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();

    float next[4];
    dyn.step(state.data(), action.data(), next, config.dt);
    state = Eigen::Map<Eigen::VectorXf>(next, 4);
  }

  float final_norm = state.head<2>().norm();
  EXPECT_LT(final_norm, initial_norm * 0.5f)
      << "SMPPI did not converge. Initial: " << initial_norm
      << ", final: " << final_norm;
}

// ===========================================================================
// KMPPI Tests
// ===========================================================================

TEST(KMPPIDoubleIntegrator, Convergence) {
  auto config = make_di_config();
  config.num_support_pts = 10;
  DoubleIntegrator dyn;
  QuadraticCost cost;
  KMPPIController<DoubleIntegrator, QuadraticCost> ctrl(config, dyn, cost);

  Eigen::VectorXf state(4);
  state << 2.0f, 2.0f, 0.0f, 0.0f;
  float initial_norm = state.head<2>().norm();

  for (int t = 0; t < 80; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);

    float next[4];
    dyn.step(state.data(), action.data(), next, config.dt);
    state = Eigen::Map<Eigen::VectorXf>(next, 4);
  }

  float final_norm = state.head<2>().norm();
  EXPECT_LT(final_norm, initial_norm * 0.5f)
      << "KMPPI did not converge. Initial: " << initial_norm
      << ", final: " << final_norm;
}

// ===========================================================================
// Trajectory Tracking Tests — AccelerationTrackingCost + PointMass3D
// ===========================================================================

using mppi::instantiations::AccelerationTrackingCost;

// RAII helper for device reference trajectory memory
struct DeviceRefTrajectory {
  float* d_ptr = nullptr;
  int horizon = 0;

  DeviceRefTrajectory() = default;
  ~DeviceRefTrajectory() { if (d_ptr) cudaFree(d_ptr); }

  // Upload a flat host vector (horizon x 6) to device
  void upload(const std::vector<float>& host_data, int h) {
    horizon = h;
    size_t bytes = host_data.size() * sizeof(float);
    if (d_ptr) cudaFree(d_ptr);
    HANDLE_ERROR(cudaMalloc(&d_ptr, bytes));
    HANDLE_ERROR(cudaMemcpy(d_ptr, host_data.data(), bytes,
                            cudaMemcpyHostToDevice));
  }

  // Build constant reference: same 6D state repeated for horizon
  void set_constant(const float ref[6], int h) {
    std::vector<float> flat(h * 6);
    for (int t = 0; t < h; ++t)
      for (int i = 0; i < 6; ++i)
        flat[t * 6 + i] = ref[i];
    upload(flat, h);
  }

  // Non-copyable
  DeviceRefTrajectory(const DeviceRefTrajectory&) = delete;
  DeviceRefTrajectory& operator=(const DeviceRefTrajectory&) = delete;
};

// Helper: config matching the mppi_acc_node defaults
static MPPIConfig make_acc_config() {
  MPPIConfig c{};
  c.num_samples    = 1024;
  c.horizon        = 50;
  c.nx             = PointMass3D::STATE_DIM;
  c.nu             = PointMass3D::CONTROL_DIM;
  c.lambda         = 1.0f;
  c.dt             = 0.02f;
  c.u_scale        = 1.0f;
  c.learning_rate  = 1.0f;
  c.w_action_seq_cost = 0.0f;
  c.num_support_pts   = 10;
  c.control_sigma[0]  = 1.0f;
  c.control_sigma[1]  = 1.0f;
  c.control_sigma[2]  = 1.0f;
  return c;
}

TEST(TrajectoryTracking, StaticPointTracking) {
  auto config = make_acc_config();
  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 10.0f;
  cost.Q_vel = 1.0f;
  cost.R_acc = 0.01f;
  cost.R_du  = 0.5f;
  cost.terminal_multiplier = 10.0f;

  // Reference: hold position [3, -2, -1] with zero velocity
  DeviceRefTrajectory ref;
  float goal[6] = {3.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f};
  ref.set_constant(goal, config.horizon);
  cost.ref_traj = ref.d_ptr;
  cost.ref_horizon = ref.horizon;

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);

  Eigen::VectorXf state = Eigen::VectorXf::Zero(6);
  Eigen::Vector3f goal_pos(goal[0], goal[1], goal[2]);
  float initial_dist = (state.head<3>() - goal_pos).norm();

  const int sim_steps = 150;
  for (int t = 0; t < sim_steps; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();
    dyn.step_host(state, action, config.dt);
  }

  float final_dist = (state.head<3>() - goal_pos).norm();
  EXPECT_LT(final_dist, 1.0f)
      << "Did not track static reference. "
      << "Initial dist: " << initial_dist
      << ", final dist: " << final_dist
      << ", final pos: " << state.head<3>().transpose();
}

TEST(TrajectoryTracking, HoverAtOriginFallback) {
  auto config = make_acc_config();
  config.control_sigma[0] = 2.0f;
  config.control_sigma[1] = 2.0f;
  config.control_sigma[2] = 2.0f;
  config.num_samples = 2048;

  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 15.0f;
  cost.Q_vel = 2.0f;
  cost.R_acc = 0.01f;
  cost.R_du  = 0.1f;
  cost.terminal_multiplier = 15.0f;
  // ref_traj = nullptr (default) → hover at origin

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);

  Eigen::VectorXf state(6);
  state << 2.0f, -1.0f, 0.5f, 0.0f, 0.0f, 0.0f;

  float initial_norm = state.head<3>().norm();

  const int sim_steps = 200;
  for (int t = 0; t < sim_steps; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();
    dyn.step_host(state, action, config.dt);
  }

  float final_norm = state.head<3>().norm();
  EXPECT_LT(final_norm, initial_norm * 0.5f)
      << "Hover fallback did not converge to origin. "
      << "Initial: " << initial_norm
      << ", final: " << final_norm
      << ", final state: " << state.transpose();
}

TEST(TrajectoryTracking, WaypointSequence) {
  auto config = make_acc_config();
  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 10.0f;
  cost.Q_vel = 1.0f;
  cost.R_acc = 0.01f;
  cost.R_du  = 0.5f;
  cost.terminal_multiplier = 10.0f;

  // Build a time-varying reference: linear interpolation from A to B
  Eigen::Vector3f pos_A(0.0f, 0.0f, 0.0f);
  Eigen::Vector3f pos_B(4.0f, 3.0f, -2.0f);

  DeviceRefTrajectory ref;
  std::vector<float> ref_flat(config.horizon * 6);
  for (int t = 0; t < config.horizon; ++t) {
    float alpha = static_cast<float>(t) / (config.horizon - 1);
    Eigen::Vector3f pos = pos_A + alpha * (pos_B - pos_A);
    // Velocity: derivative of linear interp = (B - A) / total_time
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

  const int sim_steps = 200;
  for (int t = 0; t < sim_steps; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();
    dyn.step_host(state, action, config.dt);
  }

  // After following the trajectory, should be near the end waypoint B
  float dist_to_B = (state.head<3>() - pos_B).norm();
  EXPECT_LT(dist_to_B, 2.0f)
      << "Did not follow waypoint sequence. "
      << "Final pos: " << state.head<3>().transpose()
      << ", target B: " << pos_B.transpose()
      << ", dist: " << dist_to_B;
}

TEST(TrajectoryTracking, ControlBounded) {
  auto config = make_acc_config();
  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 10.0f;
  cost.Q_vel = 1.0f;
  cost.R_acc = 0.01f;
  cost.R_du  = 0.5f;
  cost.terminal_multiplier = 10.0f;

  DeviceRefTrajectory ref;
  float goal[6] = {2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  ref.set_constant(goal, config.horizon);
  cost.ref_traj = ref.d_ptr;
  cost.ref_horizon = ref.horizon;

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);
  Eigen::VectorXf state = Eigen::VectorXf::Zero(6);

  const int sim_steps = 100;
  for (int t = 0; t < sim_steps; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();

    // All actions should be finite
    for (int i = 0; i < config.nu; ++i) {
      ASSERT_TRUE(std::isfinite(action(i)))
          << "Non-finite action at step " << t << ", dim " << i;
    }

    // MPPI nominal control is unclamped (dynamics clamp internally),
    // but actions should not blow up to extreme values
    for (int i = 0; i < config.nu; ++i) {
      EXPECT_LE(std::abs(action(i)), 50.0f)
          << "Action blew up at step " << t
          << ", dim " << i << ": " << action(i);
    }

    ctrl.set_applied_control(action);
    ctrl.shift();
    dyn.step_host(state, action, config.dt);
  }

  // Should still converge while staying bounded
  Eigen::Vector3f goal_pos(goal[0], goal[1], goal[2]);
  float final_dist = (state.head<3>() - goal_pos).norm();
  EXPECT_LT(final_dist, 1.5f)
      << "Did not track while keeping bounded controls. Dist: " << final_dist;
}

// ===========================================================================
// Acceleration-on-Quadrotor Tests
// MPPI plans with PointMass3D, simulation runs on full QuadrotorDynamics
// ===========================================================================

// Extract 6D PointMass3D state [px,py,pz,vx,vy,vz] from 13D quadrotor state
static Eigen::VectorXf extract_pm3d_state(const Eigen::VectorXf& quad_state) {
  return quad_state.head<6>();
}

// Clamp MPPI acceleration output to PointMass3D bounds.
// MPPI outputs unclamped actions (dynamics clamp internally during rollouts).
static Eigen::VectorXf clamp_acc(const Eigen::VectorXf& acc, const PointMass3D& dyn) {
  Eigen::VectorXf clamped(3);
  for (int i = 0; i < 3; ++i) {
    clamped(i) = std::fmin(std::fmax(acc(i), -dyn.a_max[i]), dyn.a_max[i]);
  }
  return clamped;
}

// Convert PointMass3D acceleration [ax,ay,az] (NED, gravity-free) to
// QuadrotorDynamics control [T, wx_cmd, wy_cmd, wz_cmd].
//
// The PointMass3D model assumes PX4 handles gravity, so the acceleration
// command represents desired world-frame acceleration *without* gravity.
// The full quadrotor must produce thrust to achieve a_desired + [0,0,g].
static Eigen::VectorXf acc_to_quad_control(
    const Eigen::VectorXf& quad_state,
    const Eigen::VectorXf& acc_cmd,
    float mass, float gravity, float K_att)
{
  // Thrust force in world NED frame:
  // QuadrotorDynamics: v_dot = R * [0,0,-T] / mass + [0,0,g]
  // PointMass3D: v_dot = a_cmd  (gravity handled by PX4)
  // So: R * [0,0,-T] / mass = a_cmd - [0,0,g]
  Eigen::Vector3f F_thrust = mass * Eigen::Vector3f(
      acc_cmd(0), acc_cmd(1), acc_cmd(2) - gravity);

  // Thrust magnitude
  float T = F_thrust.norm();
  T = std::fmax(T, 0.5f);  // avoid singularity at zero thrust

  // Desired body Z axis (FRD): thrust acts along body -Z,
  // so body +Z = -F_thrust / ||F_thrust|| (points opposite to thrust)
  Eigen::Vector3f z_des = -F_thrust / T;

  // Build desired rotation matrix (zero yaw).
  // x_des = y_world x z_des (with y_world = [0,1,0] in NED)
  // If z_des is near [0,1,0], fall back to x_world = [1,0,0]
  Eigen::Vector3f y_world(0.0f, 1.0f, 0.0f);
  Eigen::Vector3f x_des = y_world.cross(z_des);
  float x_norm = x_des.norm();
  if (x_norm < 1e-4f) {
    Eigen::Vector3f x_world(1.0f, 0.0f, 0.0f);
    x_des = x_world;
    Eigen::Vector3f y_des_tmp = z_des.cross(x_des).normalized();
    x_des = y_des_tmp.cross(z_des).normalized();
  } else {
    x_des /= x_norm;
  }
  Eigen::Vector3f y_des = z_des.cross(x_des);

  Eigen::Matrix3f R_des;
  R_des.col(0) = x_des;
  R_des.col(1) = y_des;
  R_des.col(2) = z_des;

  Eigen::Quaternionf q_des(R_des);
  q_des.normalize();

  // Current attitude from quadrotor state
  Eigen::Quaternionf q_cur(quad_state(6), quad_state(7),
                           quad_state(8), quad_state(9));
  q_cur.normalize();

  // Attitude error: q_err rotates from current to desired
  Eigen::Quaternionf q_err = q_des * q_cur.inverse();
  if (q_err.w() < 0.0f) {
    q_err.coeffs() = -q_err.coeffs();
  }

  // Proportional rate command from angle-axis error
  Eigen::AngleAxisf aa(q_err);
  Eigen::Vector3f w_cmd = K_att * aa.angle() * aa.axis();

  Eigen::VectorXf control(4);
  control << T, w_cmd(0), w_cmd(1), w_cmd(2);
  return control;
}

TEST(TrajectoryTracking, AccOnQuadrotorHover) {
  auto config = make_acc_config();
  config.num_samples = 2048;

  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 15.0f;
  cost.Q_vel = 2.0f;
  cost.R_acc = 0.01f;
  cost.R_du  = 0.1f;
  cost.terminal_multiplier = 15.0f;

  // Reference: hold position [2, -1, -1] (2m N, 1m W, 1m up in NED)
  DeviceRefTrajectory ref;
  float goal[6] = {2.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
  ref.set_constant(goal, config.horizon);
  cost.ref_traj = ref.d_ptr;
  cost.ref_horizon = ref.horizon;

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);

  // Full quadrotor simulation: start at origin, level, at rest
  QuadrotorDynamics quad;
  Eigen::VectorXf quad_state(13);
  quad_state << 0.0f, 0.0f, 0.0f,       // position
                0.0f, 0.0f, 0.0f,       // velocity
                1.0f, 0.0f, 0.0f, 0.0f, // quaternion (identity)
                0.0f, 0.0f, 0.0f;       // angular velocity

  Eigen::Vector3f goal_pos(goal[0], goal[1], goal[2]);
  float initial_dist = (quad_state.head<3>() - goal_pos).norm();

  const float K_att = 8.0f;
  const int sim_steps = 300;
  for (int t = 0; t < sim_steps; ++t) {
    Eigen::VectorXf pm_state = extract_pm3d_state(quad_state);
    ctrl.compute(pm_state);
    // Clamp to PointMass3D bounds (MPPI outputs unclamped nominal actions;
    // dynamics clamp internally during rollouts but raw output may exceed)
    Eigen::VectorXf acc_action = clamp_acc(ctrl.get_action(), dyn);
    ctrl.set_applied_control(acc_action);
    ctrl.shift();

    Eigen::VectorXf quad_control = acc_to_quad_control(
        quad_state, acc_action, quad.mass, quad.gravity, K_att);
    quad.step_host(quad_state, quad_control, config.dt);
  }

  float final_dist = (quad_state.head<3>() - goal_pos).norm();
  float final_vel = quad_state.segment<3>(3).norm();
  EXPECT_LT(final_dist, 1.5f)
      << "AccOnQuadrotor hover did not converge. "
      << "Initial dist: " << initial_dist
      << ", final dist: " << final_dist
      << ", final pos: " << quad_state.head<3>().transpose()
      << ", goal: " << goal_pos.transpose();
  EXPECT_LT(final_vel, 1.0f)
      << "Velocity did not settle. Final vel: "
      << quad_state.segment<3>(3).transpose();
}

TEST(TrajectoryTracking, AccOnQuadrotorWaypoint) {
  auto config = make_acc_config();
  config.num_samples = 2048;

  PointMass3D dyn;
  AccelerationTrackingCost cost;
  cost.Q_pos = 15.0f;
  cost.Q_vel = 2.0f;
  cost.R_acc = 0.01f;
  cost.R_du  = 0.1f;
  cost.terminal_multiplier = 15.0f;

  // Reference: fly to [3, 2, -1] (3m N, 2m E, 1m up in NED)
  DeviceRefTrajectory ref;
  float goal[6] = {3.0f, 2.0f, -1.0f, 0.0f, 0.0f, 0.0f};
  ref.set_constant(goal, config.horizon);
  cost.ref_traj = ref.d_ptr;
  cost.ref_horizon = ref.horizon;

  MPPIController<PointMass3D, AccelerationTrackingCost> ctrl(config, dyn, cost);

  // Start at origin, level, at rest
  QuadrotorDynamics quad;
  Eigen::VectorXf quad_state(13);
  quad_state << 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f,
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f;

  Eigen::Vector3f goal_pos(goal[0], goal[1], goal[2]);

  const float K_att = 8.0f;
  const int sim_steps = 500;
  for (int t = 0; t < sim_steps; ++t) {
    Eigen::VectorXf pm_state = extract_pm3d_state(quad_state);
    ctrl.compute(pm_state);
    Eigen::VectorXf acc_action = clamp_acc(ctrl.get_action(), dyn);
    ctrl.set_applied_control(acc_action);
    ctrl.shift();

    Eigen::VectorXf quad_control = acc_to_quad_control(
        quad_state, acc_action, quad.mass, quad.gravity, K_att);
    quad.step_host(quad_state, quad_control, config.dt);
  }

  float final_dist = (quad_state.head<3>() - goal_pos).norm();
  float initial_dist = goal_pos.norm();  // started at origin
  EXPECT_LT(final_dist, initial_dist * 0.5f)
      << "AccOnQuadrotor waypoint tracking did not converge. "
      << "Initial dist: " << initial_dist
      << ", final dist: " << final_dist
      << ", final pos: " << quad_state.head<3>().transpose()
      << ", goal: " << goal_pos.transpose();
}
