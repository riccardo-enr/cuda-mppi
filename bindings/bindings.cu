#define NB_CUDA
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h> // Enable Eigen <-> Numpy conversion

#include "mppi/core/mppi_common.cuh"
#include "mppi/core/map.cuh"
#include "mppi/core/fsmi.cuh"
#include "mppi/controllers/mppi.cuh"
#include "mppi/controllers/i_mppi.cuh"
// Disabled due to Eigen/CUDA compatibility issues
// #include "mppi/controllers/smppi.cuh"
// #include "mppi/controllers/kmppi.cuh"
#include "mppi/controllers/jit_mppi.hpp"
#include "mppi/instantiations/double_integrator.cuh"
#include "mppi/instantiations/quadrotor.cuh"
#include "mppi/instantiations/informative_cost.cuh"
#include "mppi/planning/trajectory_generator.hpp"

namespace nb = nanobind;
using namespace mppi;

// ---------------------------------------------------------------------------
// Helper: manages OccupancyGrid2D with device memory from Python
// ---------------------------------------------------------------------------
struct PyOccupancyGrid2D {
    OccupancyGrid2D grid;
    float* d_data = nullptr;
    int width = 0, height = 0;

    PyOccupancyGrid2D(int w, int h, float res, float ox, float oy)
        : width(w), height(h) {
        cudaMalloc(&d_data, w * h * sizeof(float));
        cudaMemset(d_data, 0, w * h * sizeof(float));
        grid.data = d_data;
        grid.dims = make_int2(w, h);
        grid.resolution = res;
        grid.origin = make_float2(ox, oy);
    }

    ~PyOccupancyGrid2D() {
        if (d_data) cudaFree(d_data);
    }

    void upload(const Eigen::VectorXf& data) {
        if (data.size() != width * height) {
            throw std::runtime_error("Data size mismatch");
        }
        cudaMemcpy(d_data, data.data(), width * height * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    Eigen::VectorXf download() const {
        Eigen::VectorXf out(width * height);
        cudaMemcpy(out.data(), d_data, width * height * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return out;
    }

    void update_fov(float uav_x, float uav_y, float yaw,
                    float fov_rad, float max_range, int n_rays, float ray_step) {
        fov_grid_update(grid, make_float2(uav_x, uav_y), yaw,
                        fov_rad, max_range, n_rays, ray_step);
    }
};

// ---------------------------------------------------------------------------
// Helper: manages InfoField from Python
// ---------------------------------------------------------------------------
struct PyInfoField {
    InfoField field;

    void compute(PyOccupancyGrid2D& py_grid, float uav_x, float uav_y,
                 float field_res, float field_extent, int n_yaw,
                 int num_beams, float max_range, float ray_step) {
        InfoFieldConfig ifc;
        ifc.field_res = field_res;
        ifc.field_extent = field_extent;
        ifc.n_yaw = n_yaw;

        FSMIConfig fsmi_cfg;
        fsmi_cfg.num_beams = num_beams;
        fsmi_cfg.max_range = max_range;
        fsmi_cfg.ray_step = ray_step;

        field.compute(py_grid.grid, make_float2(uav_x, uav_y), ifc, fsmi_cfg);
    }

    Eigen::VectorXf download() const {
        Eigen::VectorXf out(field.Nx * field.Ny);
        field.download(out.data());
        return out;
    }

    int get_Nx() const { return field.Nx; }
    int get_Ny() const { return field.Ny; }
    float get_origin_x() const { return field.origin.x; }
    float get_origin_y() const { return field.origin.y; }
    float get_res() const { return field.res; }

    ~PyInfoField() { field.free(); }
};

// ---------------------------------------------------------------------------
// Quadrotor I-MPPI controller wrapper
// ---------------------------------------------------------------------------
using QuadIMPPI = IMPPIController<instantiations::QuadrotorDynamics,
                                   instantiations::InformativeCost>;

struct PyQuadrotorIMPPI {
    QuadIMPPI controller;
    float* d_ref_traj = nullptr;
    int ref_horizon = 0;

    PyQuadrotorIMPPI(const MPPIConfig& config,
                     const instantiations::QuadrotorDynamics& dyn,
                     const instantiations::InformativeCost& cost)
        : controller(config, dyn, cost) {}

    ~PyQuadrotorIMPPI() {
        if (d_ref_traj) cudaFree(d_ref_traj);
    }

    void compute(const Eigen::VectorXf& state) {
        controller.compute(state);
    }

    Eigen::VectorXf get_action() {
        return controller.get_action();
    }

    void shift() {
        controller.shift();
    }

    void set_reference_trajectory(const Eigen::VectorXf& u_ref) {
        controller.set_reference_trajectory(u_ref);
    }

    void set_position_reference(const Eigen::VectorXf& pos_ref_flat, int horizon) {
        // Upload position reference (horizon × 3) to device
        if (d_ref_traj == nullptr || ref_horizon != horizon) {
            if (d_ref_traj) cudaFree(d_ref_traj);
            cudaMalloc(&d_ref_traj, horizon * 3 * sizeof(float));
            ref_horizon = horizon;
        }
        cudaMemcpy(d_ref_traj, pos_ref_flat.data(),
                   horizon * 3 * sizeof(float), cudaMemcpyHostToDevice);
    }
};

// ===========================================================================
NB_MODULE(cuda_mppi, m) {
    // 1. MPPIConfig (extended with I-MPPI fields)
    nb::class_<MPPIConfig>(m, "MPPIConfig")
        .def(nb::init<>())
        .def("__init__",
             [](MPPIConfig* self, int num_samples, int horizon, int nx, int nu,
                float lambda_, float dt, float u_scale, float w_action_seq_cost,
                int num_support_pts, float lambda_info, float alpha) {
                 new (self) MPPIConfig{num_samples, horizon, nx, nu,
                                       lambda_, dt, u_scale, w_action_seq_cost,
                                       num_support_pts, lambda_info, alpha};
             },
             nb::arg("num_samples"), nb::arg("horizon"),
             nb::arg("nx"), nb::arg("nu"),
             nb::arg("lambda_"), nb::arg("dt"),
             nb::arg("u_scale") = 1.0f,
             nb::arg("w_action_seq_cost") = 0.0f,
             nb::arg("num_support_pts") = 0,
             nb::arg("lambda_info") = 0.0f,
             nb::arg("alpha") = 0.0f)
        .def_rw("num_samples", &MPPIConfig::num_samples)
        .def_rw("horizon", &MPPIConfig::horizon)
        .def_rw("nx", &MPPIConfig::nx)
        .def_rw("nu", &MPPIConfig::nu)
        .def_rw("lambda_", &MPPIConfig::lambda)
        .def_rw("dt", &MPPIConfig::dt)
        .def_rw("u_scale", &MPPIConfig::u_scale)
        .def_rw("w_action_seq_cost", &MPPIConfig::w_action_seq_cost)
        .def_rw("num_support_pts", &MPPIConfig::num_support_pts)
        .def_rw("lambda_info", &MPPIConfig::lambda_info)
        .def_rw("alpha", &MPPIConfig::alpha)
        .def("__repr__", [](const MPPIConfig &c) {
            return "MPPIConfig(K=" + std::to_string(c.num_samples) +
                   ", T=" + std::to_string(c.horizon) +
                   ", nx=" + std::to_string(c.nx) +
                   ", nu=" + std::to_string(c.nu) +
                   ", alpha=" + std::to_string(c.alpha) + ")";
        });

    // 2. Original dynamics/cost instantiations
    nb::class_<instantiations::DoubleIntegrator>(m, "DoubleIntegrator")
        .def(nb::init<>());

    nb::class_<instantiations::QuadraticCost>(m, "QuadraticCost")
        .def(nb::init<>());

    using DIMPPI = MPPIController<instantiations::DoubleIntegrator, instantiations::QuadraticCost>;

    nb::class_<DIMPPI>(m, "DoubleIntegratorMPPI")
        .def(nb::init<const MPPIConfig&, const instantiations::DoubleIntegrator&, const instantiations::QuadraticCost&>(),
             nb::arg("config"),
             nb::arg("dynamics") = instantiations::DoubleIntegrator(),
             nb::arg("cost") = instantiations::QuadraticCost())
        .def("compute", &DIMPPI::compute, nb::arg("state"))
        .def("get_action", &DIMPPI::get_action)
        .def("shift", &DIMPPI::shift);

    // JIT MPPI Controller
    nb::class_<JITMPPIController>(m, "JITMPPIController")
        .def("__init__",
             [](JITMPPIController* self, const MPPIConfig& config,
                const std::string& dynamics_code, const std::string& cost_code,
                const std::vector<std::string>& include_paths) {
                 new (self) JITMPPIController(config, dynamics_code, cost_code, include_paths);
             },
             nb::arg("config"),
             nb::arg("dynamics_code"),
             nb::arg("cost_code"),
             nb::arg("include_paths"))
        .def("compute", &JITMPPIController::compute, nb::arg("state"))
        .def("get_action", &JITMPPIController::get_action)
        .def("shift", &JITMPPIController::shift)
        .def("get_nominal_trajectory", &JITMPPIController::get_nominal_trajectory);

    // ===================================================================
    // NEW: I-MPPI Feature Parity Bindings
    // ===================================================================

    // 3. OccupancyGrid2D (with device memory management)
    nb::class_<PyOccupancyGrid2D>(m, "OccupancyGrid2D")
        .def(nb::init<int, int, float, float, float>(),
             nb::arg("width"), nb::arg("height"),
             nb::arg("resolution"), nb::arg("origin_x"), nb::arg("origin_y"))
        .def("upload", &PyOccupancyGrid2D::upload, nb::arg("data"),
             "Upload flat numpy array (H*W) to GPU")
        .def("download", &PyOccupancyGrid2D::download,
             "Download grid data to numpy")
        .def("update_fov", &PyOccupancyGrid2D::update_fov,
             nb::arg("uav_x"), nb::arg("uav_y"), nb::arg("yaw"),
             nb::arg("fov_rad") = 1.57f, nb::arg("max_range") = 2.5f,
             nb::arg("n_rays") = 64, nb::arg("ray_step") = 0.1f,
             "Run FOV grid update kernel")
        .def_ro("width", &PyOccupancyGrid2D::width)
        .def_ro("height", &PyOccupancyGrid2D::height);

    // 4. InfoField
    nb::class_<PyInfoField>(m, "InfoField")
        .def(nb::init<>())
        .def("compute", &PyInfoField::compute,
             nb::arg("grid"), nb::arg("uav_x"), nb::arg("uav_y"),
             nb::arg("field_res") = 0.5f, nb::arg("field_extent") = 5.0f,
             nb::arg("n_yaw") = 8,
             nb::arg("num_beams") = 12, nb::arg("max_range") = 10.0f,
             nb::arg("ray_step") = 0.1f,
             "Compute information field centred on UAV")
        .def("download", &PyInfoField::download, "Download field to numpy")
        .def_prop_ro("Nx", &PyInfoField::get_Nx)
        .def_prop_ro("Ny", &PyInfoField::get_Ny)
        .def_prop_ro("origin_x", &PyInfoField::get_origin_x)
        .def_prop_ro("origin_y", &PyInfoField::get_origin_y)
        .def_prop_ro("res", &PyInfoField::get_res);

    // 5. QuadrotorDynamics
    nb::class_<instantiations::QuadrotorDynamics>(m, "QuadrotorDynamics")
        .def(nb::init<>())
        .def_rw("mass", &instantiations::QuadrotorDynamics::mass)
        .def_rw("gravity", &instantiations::QuadrotorDynamics::gravity)
        .def_rw("tau_omega", &instantiations::QuadrotorDynamics::tau_omega);

    // 6. InformativeCost
    nb::class_<instantiations::InformativeCost>(m, "InformativeCost")
        .def(nb::init<>())
        .def_rw("lambda_info", &instantiations::InformativeCost::lambda_info)
        .def_rw("lambda_local", &instantiations::InformativeCost::lambda_local)
        .def_rw("target_weight", &instantiations::InformativeCost::target_weight)
        .def_rw("goal_weight", &instantiations::InformativeCost::goal_weight)
        .def_rw("collision_penalty", &instantiations::InformativeCost::collision_penalty)
        .def_rw("height_weight", &instantiations::InformativeCost::height_weight)
        .def_rw("target_altitude", &instantiations::InformativeCost::target_altitude)
        .def_rw("action_reg", &instantiations::InformativeCost::action_reg);

    // 7. QuadrotorIMPPI (wrapper)
    nb::class_<PyQuadrotorIMPPI>(m, "QuadrotorIMPPI")
        .def(nb::init<const MPPIConfig&,
                       const instantiations::QuadrotorDynamics&,
                       const instantiations::InformativeCost&>(),
             nb::arg("config"),
             nb::arg("dynamics") = instantiations::QuadrotorDynamics(),
             nb::arg("cost") = instantiations::InformativeCost())
        .def("compute", &PyQuadrotorIMPPI::compute, nb::arg("state"),
             "Run one MPPI iteration")
        .def("get_action", &PyQuadrotorIMPPI::get_action,
             "Get first control from nominal trajectory")
        .def("shift", &PyQuadrotorIMPPI::shift,
             "Shift nominal trajectory forward")
        .def("set_reference_trajectory", &PyQuadrotorIMPPI::set_reference_trajectory,
             nb::arg("u_ref"), "Set control reference for biased sampling")
        .def("set_position_reference", &PyQuadrotorIMPPI::set_position_reference,
             nb::arg("pos_ref_flat"), nb::arg("horizon"),
             "Upload position reference trajectory (horizon*3) to device");

    // 8. TrajectoryGenerator
    using TG = planning::TrajectoryGenerator;
    using TGConfig = planning::TrajectoryGeneratorConfig;
    using TGResult = TG::TargetResult;

    nb::class_<TGConfig>(m, "TrajectoryGeneratorConfig")
        .def(nb::init<>())
        .def_rw("info_threshold", &TGConfig::info_threshold)
        .def_rw("ref_speed", &TGConfig::ref_speed)
        .def_rw("dist_weight", &TGConfig::dist_weight)
        .def_rw("goal_x", &TGConfig::goal_x)
        .def_rw("goal_y", &TGConfig::goal_y)
        .def_rw("goal_z", &TGConfig::goal_z);

    nb::class_<TGResult>(m, "TargetResult")
        .def_ro("x", &TGResult::x)
        .def_ro("y", &TGResult::y)
        .def_ro("z", &TGResult::z)
        .def_ro("mode", &TGResult::mode);

    nb::class_<TG>(m, "TrajectoryGenerator")
        .def("__init__",
             [](TG* self, const TGConfig& cfg,
                const std::vector<std::vector<float>>& zones_list) {
                 std::vector<planning::InfoZone> zones;
                 for (const auto& z : zones_list) {
                     if (z.size() >= 5) {
                         zones.push_back({z[0], z[1], z[2], z[3], z[4]});
                     }
                 }
                 new (self) TG(cfg, zones);
             },
             nb::arg("config"), nb::arg("zones"))
        .def("select_target",
             [](const TG& self, float px, float py, float pz,
                const std::vector<float>& info_levels) {
                 return self.select_target(px, py, pz, info_levels);
             },
             nb::arg("px"), nb::arg("py"), nb::arg("pz"),
             nb::arg("info_levels"))
        .def("make_ref_trajectory",
             [](const TG& self, float sx, float sy, float sz,
                float tx, float ty, float tz, int horizon, float dt) {
                 auto traj = self.make_ref_trajectory(sx, sy, sz, tx, ty, tz, horizon, dt);
                 // Convert to Eigen for numpy compatibility
                 Eigen::VectorXf result(traj.size());
                 for (size_t i = 0; i < traj.size(); ++i) result(i) = traj[i];
                 return result;
             },
             nb::arg("sx"), nb::arg("sy"), nb::arg("sz"),
             nb::arg("tx"), nb::arg("ty"), nb::arg("tz"),
             nb::arg("horizon"), nb::arg("dt"))
        .def("field_gradient_trajectory",
             [](const TG& self, const Eigen::VectorXf& field,
                int Nx, int Ny, float ox, float oy, float res,
                float sx, float sy, int horizon, float speed, float dt, float alt) {
                 auto traj = self.field_gradient_trajectory(
                     field.data(), Nx, Ny, ox, oy, res, sx, sy,
                     horizon, speed, dt, alt);
                 Eigen::VectorXf result(traj.size());
                 for (size_t i = 0; i < traj.size(); ++i) result(i) = traj[i];
                 return result;
             },
             nb::arg("field"), nb::arg("Nx"), nb::arg("Ny"),
             nb::arg("ox"), nb::arg("oy"), nb::arg("res"),
             nb::arg("sx"), nb::arg("sy"),
             nb::arg("horizon"), nb::arg("speed"), nb::arg("dt"), nb::arg("alt"));
}
