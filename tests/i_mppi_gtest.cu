/*
 * i_mppi_gtest.cu — Per-component GTest suite for the I-MPPI pipeline.
 *
 * Uses DoubleIntegrator3D (6D state, 3D acceleration control) paired with
 * InformativeCost3D — the correct dynamics for an outer-loop MPPI planner
 * sitting above an SO(3) attitude controller.
 *
 * Components tested (bottom-up):
 *   1. DoubleIntegrator3D  — Euler integration, clamping, free-fall
 *   2. OccupancyGrid2D/3D  — coordinate transforms, index linearisation, OOB
 *   3. InfoField           — allocation, upload, bilinear sampling
 *   4. InformativeCost3D   — per-layer isolation, weight zeroing
 *   5. IMPPIController     — construction, reference upload, convergence
 *   6. Full pipeline       — closed-loop I-MPPI + DI3D + informative cost
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "mppi/controllers/i_mppi.cuh"
#include "mppi/instantiations/double_integrator_3d.cuh"
#include "mppi/instantiations/informative_cost_3d.cuh"
#include "mppi/core/map.cuh"
#include "mppi/core/fsmi.cuh"

using namespace mppi;
using namespace mppi::instantiations;

/* ===================================================================
 * Helpers
 * =================================================================== */

/*
 * 6D state at a given position (NED: negative z = up).
 */
static Eigen::VectorXf make_state(float px = 0.f, float py = 0.f,
                                  float pz = -2.f) {
  Eigen::VectorXf s = Eigen::VectorXf::Zero(6);
  s[0] = px;
  s[1] = py;
  s[2] = pz;
  return s;
}

/*
 * Default I-MPPI config for DoubleIntegrator3D + InformativeCost3D.
 */
static MPPIConfig make_imppi_config() {
  MPPIConfig c{};
  c.num_samples       = 512;
  c.horizon           = 50;
  c.nx                = DoubleIntegrator3D::STATE_DIM;   // 6
  c.nu                = DoubleIntegrator3D::CONTROL_DIM;  // 3
  c.lambda            = 1.0f;
  c.dt                = 0.02f;
  c.u_scale           = 1.0f;
  c.learning_rate     = 1.0f;
  c.w_action_seq_cost = 0.0f;
  c.num_support_pts   = 10;
  c.lambda_info       = 5.0f;
  c.alpha             = 0.3f;
  c.control_sigma[0]  = 2.0f;   // ax
  c.control_sigma[1]  = 2.0f;   // ay
  c.control_sigma[2]  = 2.0f;   // az
  return c;
}

/*
 * RAII helper: 2D occupancy grid on the device.
 */
struct DeviceGrid2D {
  float *d_data = nullptr;
  OccupancyGrid2D grid{};
  int W, H;

  DeviceGrid2D(int w, int h, float res, float ox = 0.f, float oy = 0.f)
      : W(w), H(h) {
    std::vector<float> host(W * H, 0.0f);
    HANDLE_ERROR(cudaMalloc(&d_data, W * H * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_data, host.data(), W * H * sizeof(float),
                            cudaMemcpyHostToDevice));
    grid.data       = d_data;
    grid.dims       = make_int2(W, H);
    grid.resolution = res;
    grid.origin     = make_float2(ox, oy);
  }

  ~DeviceGrid2D() { if (d_data) cudaFree(d_data); }

  void upload(const std::vector<float> &host) {
    HANDLE_ERROR(cudaMemcpy(d_data, host.data(), W * H * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  void set_occupied(int x0, int y0, int x1, int y1) {
    std::vector<float> host(W * H);
    HANDLE_ERROR(cudaMemcpy(host.data(), d_data, W * H * sizeof(float),
                            cudaMemcpyDeviceToHost));
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x)
        host[y * W + x] = 1.0f;
    upload(host);
  }

  DeviceGrid2D(const DeviceGrid2D &) = delete;
  DeviceGrid2D &operator=(const DeviceGrid2D &) = delete;
};

/*
 * RAII helper: device-side reference trajectory [horizon x 3].
 */
struct DeviceRefPos {
  float *d_ptr = nullptr;

  DeviceRefPos() = default;
  ~DeviceRefPos() { if (d_ptr) cudaFree(d_ptr); }

  void set_constant(float px, float py, float pz, int horizon) {
    std::vector<float> flat(horizon * 3);
    for (int t = 0; t < horizon; ++t) {
      flat[t * 3 + 0] = px;
      flat[t * 3 + 1] = py;
      flat[t * 3 + 2] = pz;
    }
    size_t bytes = flat.size() * sizeof(float);
    if (d_ptr) cudaFree(d_ptr);
    HANDLE_ERROR(cudaMalloc(&d_ptr, bytes));
    HANDLE_ERROR(cudaMemcpy(d_ptr, flat.data(), bytes, cudaMemcpyHostToDevice));
  }

  DeviceRefPos(const DeviceRefPos &) = delete;
  DeviceRefPos &operator=(const DeviceRefPos &) = delete;
};

/* ===================================================================
 * 1. DoubleIntegrator3D
 * =================================================================== */

TEST(DoubleIntegrator3D_Test, EulerIntegration) {
  DoubleIntegrator3D dyn;

  float x[6] = {1.f, 2.f, -3.f, 0.5f, -0.5f, 0.f};
  float u[3] = {1.f, 0.f, -1.f};
  float x_next[6];
  float dt = 0.1f;

  dyn.step(x, u, x_next, dt);

  /* p_next = p + v * dt */
  EXPECT_NEAR(x_next[0], 1.0f + 0.5f * 0.1f, 1e-5f);
  EXPECT_NEAR(x_next[1], 2.0f - 0.5f * 0.1f, 1e-5f);
  EXPECT_NEAR(x_next[2], -3.0f + 0.0f * 0.1f, 1e-5f);

  /* v_next = v + a * dt */
  EXPECT_NEAR(x_next[3], 0.5f + 1.0f * 0.1f, 1e-5f);
  EXPECT_NEAR(x_next[4], -0.5f + 0.0f * 0.1f, 1e-5f);
  EXPECT_NEAR(x_next[5], 0.0f - 1.0f * 0.1f, 1e-5f);
}

TEST(DoubleIntegrator3D_Test, AccelerationClamping) {
  DoubleIntegrator3D dyn;
  dyn.a_max = 3.0f;

  float x[6] = {};
  float u[3] = {10.f, -10.f, 0.f};
  float x_next[6];

  dyn.step(x, u, x_next, 1.0f);

  EXPECT_NEAR(x_next[3], 3.0f, 1e-5f);
  EXPECT_NEAR(x_next[4], -3.0f, 1e-5f);
  EXPECT_NEAR(x_next[5], 0.0f, 1e-5f);
}

TEST(DoubleIntegrator3D_Test, FreeFallAccuracy) {
  DoubleIntegrator3D dyn;
  dyn.a_max = 20.0f;

  /* Constant 1 m/s^2 downward for 1 s -> p_z += 0.5 m */
  Eigen::VectorXf state = Eigen::VectorXf::Zero(6);
  Eigen::VectorXf accel(3);
  accel << 0.f, 0.f, 1.f;

  float dt = 0.001f;
  for (int i = 0; i < 1000; ++i) {
    dyn.step_host(state, accel, dt);
  }

  EXPECT_NEAR(state[2], 0.5f, 0.01f);
  EXPECT_NEAR(state[5], 1.0f, 0.01f);
}

TEST(DoubleIntegrator3D_Test, StepHostMatchesStep) {
  DoubleIntegrator3D dyn;

  Eigen::VectorXf state_h(6);
  state_h << 1.f, 2.f, -1.f, 0.3f, -0.2f, 0.1f;
  Eigen::VectorXf action(3);
  action << 0.5f, -0.5f, 1.0f;

  float x[6], u[3], x_next[6];
  for (int i = 0; i < 6; ++i) x[i] = state_h(i);
  for (int i = 0; i < 3; ++i) u[i] = action(i);

  dyn.step(x, u, x_next, 0.05f);
  dyn.step_host(state_h, action, 0.05f);

  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(state_h(i), x_next[i]) << "mismatch at index " << i;
  }
}

/* ===================================================================
 * 2. OccupancyGrid2D
 * =================================================================== */

TEST(OccupancyGrid2D, WorldToGridRoundTrip) {
  OccupancyGrid2D grid{};
  grid.dims       = make_int2(100, 100);
  grid.resolution = 0.1f;
  grid.origin     = make_float2(-5.0f, -5.0f);

  float2 world = make_float2(2.3f, -1.7f);
  int2 cell = grid.world_to_grid(world);
  float2 centre = grid.grid_to_world(cell);

  EXPECT_NEAR(centre.x, world.x, grid.resolution);
  EXPECT_NEAR(centre.y, world.y, grid.resolution);
}

TEST(OccupancyGrid2D, IndexLinearisation) {
  OccupancyGrid2D grid{};
  grid.dims       = make_int2(50, 30);
  grid.resolution = 0.2f;
  grid.origin     = make_float2(0.f, 0.f);

  EXPECT_EQ(grid.get_index(0, 0), 0);
  EXPECT_EQ(grid.get_index(49, 0), 49);
  EXPECT_EQ(grid.get_index(0, 1), 50);
  EXPECT_EQ(grid.get_index(49, 29), 50 * 30 - 1);
  EXPECT_EQ(grid.get_index(-1, 0), -1);
  EXPECT_EQ(grid.get_index(50, 0), -1);
  EXPECT_EQ(grid.get_index(0, 30), -1);
}

/* ===================================================================
 * 2b. OccupancyGrid (3D)
 * =================================================================== */

TEST(OccupancyGrid3D, WorldToGridRoundTrip) {
  OccupancyGrid grid{};
  grid.dims       = make_int3(50, 50, 20);
  grid.resolution = 0.2f;
  grid.origin     = make_float3(0.f, 0.f, -4.f);

  float3 world = make_float3(3.5f, 7.2f, -1.8f);
  int3 cell = grid.world_to_grid(world);
  float3 centre = grid.grid_to_world(cell);

  EXPECT_NEAR(centre.x, world.x, grid.resolution);
  EXPECT_NEAR(centre.y, world.y, grid.resolution);
  EXPECT_NEAR(centre.z, world.z, grid.resolution);
}

TEST(OccupancyGrid3D, IndexLinearisation) {
  OccupancyGrid grid{};
  grid.dims       = make_int3(10, 20, 5);
  grid.resolution = 0.5f;
  grid.origin     = make_float3(0.f, 0.f, 0.f);

  EXPECT_EQ(grid.get_index(0, 0, 0), 0);
  EXPECT_EQ(grid.get_index(9, 0, 0), 9);
  EXPECT_EQ(grid.get_index(0, 1, 0), 10);
  EXPECT_EQ(grid.get_index(0, 0, 1), 200);
  EXPECT_EQ(grid.get_index(-1, 0, 0), -1);
  EXPECT_EQ(grid.get_index(10, 0, 0), -1);
}

/* ===================================================================
 * 3. InfoField
 * =================================================================== */

__global__ void sample_info_field_kernel(
    InfoField field, float2 pos, float *d_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_out = field.sample(pos);
  }
}

TEST(InfoField, AllocateAndDownload) {
  InfoField field;
  int Nx = 20, Ny = 20;
  field.allocate(Nx, Ny);
  field.res = 0.5f;
  field.origin = make_float2(0.f, 0.f);

  ASSERT_NE(field.d_field, nullptr);

  std::vector<float> host(Nx * Ny);
  field.download(host.data());
  for (int i = 0; i < Nx * Ny; ++i) {
    EXPECT_FLOAT_EQ(host[i], 0.0f);
  }
  cudaFree(field.d_field);
}

TEST(InfoField, BilinearSampleInBounds) {
  InfoField field;
  int Nx = 4, Ny = 4;
  field.allocate(Nx, Ny);
  field.res = 1.0f;
  field.origin = make_float2(0.f, 0.f);
  field.Nx = Nx;
  field.Ny = Ny;

  std::vector<float> host(Nx * Ny);
  for (int j = 0; j < Ny; ++j)
    for (int i = 0; i < Nx; ++i)
      host[j * Nx + i] = (float)(i + j);
  cudaMemcpy(field.d_field, host.data(), Nx * Ny * sizeof(float),
             cudaMemcpyHostToDevice);

  float *d_out;
  cudaMalloc(&d_out, sizeof(float));
  sample_info_field_kernel<<<1, 1>>>(field, make_float2(1.5f, 1.5f), d_out);
  cudaDeviceSynchronize();

  float result;
  cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_NEAR(result, 2.0f, 0.5f);

  cudaFree(d_out);
  cudaFree(field.d_field);
}

TEST(InfoField, OutOfBoundsReturnsZero) {
  InfoField field;
  int Nx = 4, Ny = 4;
  field.allocate(Nx, Ny);
  field.res = 1.0f;
  field.origin = make_float2(0.f, 0.f);
  field.Nx = Nx;
  field.Ny = Ny;

  std::vector<float> ones(Nx * Ny, 1.0f);
  cudaMemcpy(field.d_field, ones.data(), Nx * Ny * sizeof(float),
             cudaMemcpyHostToDevice);

  float *d_out;
  cudaMalloc(&d_out, sizeof(float));
  sample_info_field_kernel<<<1, 1>>>(field, make_float2(-10.f, -10.f), d_out);
  cudaDeviceSynchronize();

  float result;
  cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(result, 0.0f);

  cudaFree(d_out);
  cudaFree(field.d_field);
}

/* ===================================================================
 * 4. InformativeCost3D — per-layer isolation
 * =================================================================== */

__global__ void eval_cost3d_kernel(
    InformativeCost3D cost, const float *x, const float *u, int t,
    float *d_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float u_prev[3] = {};
    *d_out = cost.compute(x, u, u_prev, t);
  }
}

__global__ void eval_terminal3d_kernel(
    InformativeCost3D cost, const float *x, float *d_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_out = cost.terminal_cost(x);
  }
}

static float eval_cost(InformativeCost3D &cost, const float *h_x,
                       const float *h_u, int t) {
  float *d_x, *d_u, *d_out;
  cudaMalloc(&d_x, 6 * sizeof(float));
  cudaMalloc(&d_u, 3 * sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  cudaMemcpy(d_x, h_x, 6 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, h_u, 3 * sizeof(float), cudaMemcpyHostToDevice);

  eval_cost3d_kernel<<<1, 1>>>(cost, d_x, d_u, t, d_out);
  cudaDeviceSynchronize();

  float result;
  cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_u);
  cudaFree(d_out);
  return result;
}

static float eval_terminal(InformativeCost3D &cost, const float *h_x) {
  float *d_x, *d_out;
  cudaMalloc(&d_x, 6 * sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  cudaMemcpy(d_x, h_x, 6 * sizeof(float), cudaMemcpyHostToDevice);

  eval_terminal3d_kernel<<<1, 1>>>(cost, d_x, d_out);
  cudaDeviceSynchronize();

  float result;
  cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_out);
  return result;
}

TEST(InformativeCost3D_Test, HeightCostOnly) {
  DeviceGrid2D dg(10, 10, 1.0f);

  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.collision_penalty = 0.0f;
  cost.height_weight     = 10.0f;
  cost.target_altitude   = -2.0f;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.target_weight     = 0.0f;
  cost.action_reg        = 0.0f;
  cost.velocity_weight   = 0.0f;

  float x_ok[6] = {0, 0, -2, 0, 0, 0};
  float u[3] = {};
  EXPECT_NEAR(eval_cost(cost, x_ok, u, 0), 0.0f, 0.1f);

  /* 2 m off target → 10 * (0 - (-2))^2 = 40 */
  float x_off[6] = {0, 0, 0, 0, 0, 0};
  EXPECT_NEAR(eval_cost(cost, x_off, u, 0), 40.0f, 1.0f);
}

TEST(InformativeCost3D_Test, CollisionPenalty) {
  DeviceGrid2D dg(10, 10, 1.0f);
  std::vector<float> occ(100, 1.0f);
  dg.upload(occ);

  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.collision_penalty = 500.0f;
  cost.occ_threshold     = 0.7f;
  cost.height_weight     = 0.0f;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.target_weight     = 0.0f;
  cost.action_reg        = 0.0f;
  cost.velocity_weight   = 0.0f;
  cost.bound_x_min       = -100.f;
  cost.bound_x_max       = 100.f;
  cost.bound_y_min       = -100.f;
  cost.bound_y_max       = 100.f;

  float x[6] = {5, 5, 0, 0, 0, 0};
  float u[3] = {};
  EXPECT_GE(eval_cost(cost, x, u, 0), 500.0f);
}

TEST(InformativeCost3D_Test, BoundsPenalty) {
  DeviceGrid2D dg(10, 10, 1.0f);

  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.collision_penalty = 1000.0f;
  cost.height_weight     = 0.0f;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.target_weight     = 0.0f;
  cost.action_reg        = 0.0f;
  cost.velocity_weight   = 0.0f;
  cost.bound_x_min       = 0.0f;
  cost.bound_x_max       = 10.0f;
  cost.bound_y_min       = 0.0f;
  cost.bound_y_max       = 10.0f;

  float x_in[6]  = {5, 5, 0, 0, 0, 0};
  float x_out[6] = {-5, 5, 0, 0, 0, 0};
  float u[3] = {};

  float c_in  = eval_cost(cost, x_in, u, 0);
  float c_out = eval_cost(cost, x_out, u, 0);
  EXPECT_GT(c_out, c_in + 500.f);
}

TEST(InformativeCost3D_Test, ActionRegularisation) {
  DeviceGrid2D dg(10, 10, 1.0f);

  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.collision_penalty = 0.0f;
  cost.height_weight     = 0.0f;
  cost.target_altitude   = 0.0f;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.target_weight     = 0.0f;
  cost.action_reg        = 1.0f;
  cost.velocity_weight   = 0.0f;
  cost.bound_x_min       = -100.f;
  cost.bound_x_max       = 100.f;
  cost.bound_y_min       = -100.f;
  cost.bound_y_max       = 100.f;

  float x[6] = {};
  float u_zero[3] = {};
  EXPECT_NEAR(eval_cost(cost, x, u_zero, 0), 0.0f, 0.1f);

  /* action_reg * (4 + 9 + 1) = 14 */
  float u_big[3] = {2.0f, 3.0f, -1.0f};
  EXPECT_NEAR(eval_cost(cost, x, u_big, 0), 14.0f, 0.5f);
}

TEST(InformativeCost3D_Test, TerminalCost) {
  DeviceGrid2D dg(10, 10, 1.0f);

  InformativeCost3D cost{};
  cost.grid            = dg.grid;
  cost.height_weight   = 5.0f;
  cost.target_altitude = -2.0f;
  cost.target_weight   = 0.0f;

  /* 2 m error → 5 * 4 = 20 */
  float x[6] = {0, 0, 0, 0, 0, 0};
  float tc = eval_terminal(cost, x);
  EXPECT_NEAR(tc, 20.0f, 1.0f);
}

TEST(InformativeCost3D_Test, RefTrajectoryTracking) {
  DeviceGrid2D dg(100, 100, 0.1f);
  DeviceRefPos ref;
  ref.set_constant(5.0f, 5.0f, -2.0f, 50);

  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.collision_penalty = 0.0f;
  cost.height_weight     = 0.0f;
  cost.target_altitude   = 0.0f;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.target_weight     = 10.0f;
  cost.action_reg        = 0.0f;
  cost.velocity_weight   = 0.0f;
  cost.bound_x_min       = -100.f;
  cost.bound_x_max       = 100.f;
  cost.bound_y_min       = -100.f;
  cost.bound_y_max       = 100.f;
  cost.ref_trajectory    = ref.d_ptr;
  cost.ref_horizon       = 50;

  float x_on[6] = {5, 5, -2, 0, 0, 0};
  float u[3] = {};
  EXPECT_NEAR(eval_cost(cost, x_on, u, 0), 0.0f, 0.5f);

  /* 5 m away → target_weight * 5.0 = 50 */
  float x_off[6] = {0, 5, -2, 0, 0, 0};
  EXPECT_NEAR(eval_cost(cost, x_off, u, 0), 50.0f, 2.0f);
}

/* ===================================================================
 * 5. IMPPIController
 * =================================================================== */

TEST(IMPPIController, Construction) {
  auto config = make_imppi_config();
  DoubleIntegrator3D dyn;
  InformativeCost3D cost{};

  DeviceGrid2D dg(10, 10, 1.0f);
  cost.grid = dg.grid;

  IMPPIController<DoubleIntegrator3D, InformativeCost3D> ctrl(config, dyn, cost);
}

TEST(IMPPIController, SingleComputeFinite) {
  auto config = make_imppi_config();
  DoubleIntegrator3D dyn;

  DeviceGrid2D dg(100, 100, 0.1f);
  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.collision_penalty = 0.0f;
  cost.bound_x_min       = -100.f;
  cost.bound_x_max       = 100.f;
  cost.bound_y_min       = -100.f;
  cost.bound_y_max       = 100.f;

  IMPPIController<DoubleIntegrator3D, InformativeCost3D> ctrl(config, dyn, cost);

  Eigen::VectorXf state = make_state();
  ctrl.compute(state);
  Eigen::VectorXf action = ctrl.get_action();

  ASSERT_EQ(action.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(std::isfinite(action(i)))
        << "action(" << i << ") = " << action(i);
  }
}

TEST(IMPPIController, ReferenceTrajectoryUpload) {
  auto config = make_imppi_config();
  DoubleIntegrator3D dyn;

  DeviceGrid2D dg(10, 10, 1.0f);
  InformativeCost3D cost{};
  cost.grid = dg.grid;

  IMPPIController<DoubleIntegrator3D, InformativeCost3D> ctrl(config, dyn, cost);

  Eigen::VectorXf u_ref = Eigen::VectorXf::Ones(config.horizon * config.nu);
  ctrl.set_reference_trajectory(u_ref);

  Eigen::VectorXf state = make_state();
  ctrl.compute(state);
  Eigen::VectorXf action = ctrl.get_action();
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(std::isfinite(action(i)));
  }
}

/* ===================================================================
 * 6. Full pipeline — closed-loop I-MPPI + DoubleIntegrator3D
 * =================================================================== */

TEST(IMPPIPipeline, AltitudeConvergence) {
  auto config = make_imppi_config();
  config.num_samples = 2048;
  config.horizon     = 64;
  config.lambda      = 0.5f;
  DoubleIntegrator3D dyn;
  dyn.a_max = 5.0f;

  DeviceGrid2D dg(100, 100, 0.1f);
  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.height_weight     = 10.0f;
  cost.target_altitude   = -2.0f;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.target_weight     = 0.0f;
  cost.collision_penalty = 0.0f;
  cost.action_reg        = 0.5f;     // stronger regularisation to damp oscillations
  cost.velocity_weight   = 2.0f;     // penalise high speed to prevent overshoot
  cost.max_velocity      = 2.0f;
  cost.bound_x_min       = -100.f;
  cost.bound_x_max       = 100.f;
  cost.bound_y_min       = -100.f;
  cost.bound_y_max       = 100.f;

  IMPPIController<DoubleIntegrator3D, InformativeCost3D> ctrl(config, dyn, cost);

  /* Start 2 m below target altitude (pz = 0, target = -2). */
  Eigen::VectorXf state = make_state(0.f, 0.f, 0.f);
  float initial_err = std::abs(state[2] - cost.target_altitude);

  for (int t = 0; t < 300; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();
    dyn.step_host(state, action, config.dt);
  }

  float final_err = std::abs(state[2] - cost.target_altitude);
  EXPECT_LT(final_err, initial_err * 0.5f)
      << "Initial err: " << initial_err
      << ", final err: " << final_err
      << ", final pz: " << state[2];
}

TEST(IMPPIPipeline, ObstacleAvoidance) {
  auto config = make_imppi_config();
  config.num_samples = 2048;
  config.horizon     = 64;
  config.lambda      = 0.5f;
  DoubleIntegrator3D dyn;
  dyn.a_max = 5.0f;

  /* 20x20 m grid at 0.1 m/cell. Wall at y in [8, 12] m (thick). */
  int W = 200, H = 200;
  DeviceGrid2D dg(W, H, 0.1f);
  dg.set_occupied(0, 80, W, 120);

  InformativeCost3D cost{};
  cost.grid              = dg.grid;
  cost.collision_penalty = 5000.0f;
  cost.occ_threshold     = 0.7f;
  cost.height_weight     = 0.0f;    // ignore height for this test
  cost.target_altitude   = 0.0f;
  cost.lambda_local      = 0.0f;
  cost.lambda_info       = 0.0f;
  cost.target_weight     = 0.0f;
  cost.action_reg        = 0.1f;
  cost.velocity_weight   = 2.0f;
  cost.max_velocity      = 2.0f;
  cost.bound_x_min       = -1.0f;
  cost.bound_x_max       = 19.0f;
  cost.bound_y_min       = -1.0f;
  cost.bound_y_max       = 19.0f;

  IMPPIController<DoubleIntegrator3D, InformativeCost3D> ctrl(config, dyn, cost);

  /* Start at y=5, heading toward wall at y=8..12. Apply forward bias. */
  Eigen::VectorXf state = make_state(10.f, 5.f, 0.f);
  state[4] = 1.5f;   // initial velocity toward wall

  bool entered_wall = false;
  for (int t = 0; t < 200; ++t) {
    ctrl.compute(state);
    Eigen::VectorXf action = ctrl.get_action();
    ctrl.set_applied_control(action);
    ctrl.shift();
    dyn.step_host(state, action, config.dt);

    if (state[1] >= 8.0f && state[1] <= 12.0f) {
      entered_wall = true;
    }
    if (!std::isfinite(state[0])) break;
  }

  EXPECT_FALSE(entered_wall)
      << "UAV entered obstacle region. Final pos: ("
      << state[0] << ", " << state[1] << ", " << state[2] << ")";
}
