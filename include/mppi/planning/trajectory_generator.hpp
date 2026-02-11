#ifndef MPPI_TRAJECTORY_GENERATOR_HPP
#define MPPI_TRAJECTORY_GENERATOR_HPP

#include <vector>
#include <cmath>
#include <algorithm>


namespace mppi {
namespace planning {

// ---------------------------------------------------------------------------
// Information zone descriptor (matches Python INFO_ZONES format)
// [cx, cy, width, height, initial_value]
// ---------------------------------------------------------------------------
struct InfoZone {
    float cx, cy;        // centre
    float width, height; // extent
    float initial_value;
};

// ---------------------------------------------------------------------------
// Configuration for trajectory generator (Layer 2)
// ---------------------------------------------------------------------------
struct TrajectoryGeneratorConfig {
    float info_threshold = 20.0f;
    float ref_speed      = 2.0f;
    float dist_weight    = 0.5f;
    float goal_x = 9.0f, goal_y = 5.0f, goal_z = -2.0f;
};

// ---------------------------------------------------------------------------
// TrajectoryGenerator (host-side, runs at 5 Hz)
//
// Ported from fsmi.py::FSMITrajectoryGenerator
// ---------------------------------------------------------------------------
class TrajectoryGenerator {
public:
    TrajectoryGeneratorConfig config;
    std::vector<InfoZone> info_zones;

    TrajectoryGenerator() = default;
    TrajectoryGenerator(const TrajectoryGeneratorConfig& cfg,
                        const std::vector<InfoZone>& zones)
        : config(cfg), info_zones(zones) {}

    // Select next target: zone with max(info_level − dist_weight * distance)
    // Returns: (target_x, target_y, target_z, mode)
    //   mode 0 = goal, mode 1+ = zone index + 1
    struct TargetResult {
        float x, y, z;
        int mode;
    };

    TargetResult select_target(
        float px, float py, float pz,
        const std::vector<float>& info_levels
    ) const {
        int N = (int)info_zones.size();
        float best_score = -1e30f;
        int best_idx = -1;

        for (int i = 0; i < N; ++i) {
            if (info_levels[i] <= config.info_threshold) continue;
            float dx = info_zones[i].cx - px;
            float dy = info_zones[i].cy - py;
            float dist = sqrtf(dx*dx + dy*dy);
            float score = info_levels[i] - config.dist_weight * dist;
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        TargetResult result;
        if (best_idx >= 0) {
            result.x = info_zones[best_idx].cx;
            result.y = info_zones[best_idx].cy;
            result.z = config.goal_z;
            result.mode = best_idx + 1;
        } else {
            // All zones depleted → go to goal
            result.x = config.goal_x;
            result.y = config.goal_y;
            result.z = config.goal_z;
            result.mode = 0;
        }
        return result;
    }

    // Generate straight-line reference trajectory at ref_speed
    // Returns flat array: [x0,y0,z0, x1,y1,z1, ...]
    std::vector<float> make_ref_trajectory(
        float sx, float sy, float sz,
        float tx, float ty, float tz,
        int horizon, float dt
    ) const {
        float dx = tx - sx, dy = ty - sy, dz = tz - sz;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-6f;
        float ux = dx / dist, uy = dy / dist, uz = dz / dist;

        std::vector<float> traj(horizon * 3);
        for (int i = 0; i < horizon; ++i) {
            float d = config.ref_speed * dt * (float)i;
            if (d > dist) d = dist;
            traj[i*3 + 0] = sx + d * ux;
            traj[i*3 + 1] = sy + d * uy;
            traj[i*3 + 2] = sz + d * uz;
        }
        return traj;
    }

    // Bilinear interpolation on host-side field
    static float interp2d(
        const float* field, int Nx, int Ny,
        float field_ox, float field_oy, float field_res,
        float wx, float wy
    ) {
        float fx = (wx - field_ox) / field_res - 0.5f;
        float fy = (wy - field_oy) / field_res - 0.5f;

        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        float sx = fx - (float)x0;
        float sy = fy - (float)y0;

        auto clamp = [](int v, int lo, int hi) {
            return std::max(lo, std::min(v, hi));
        };
        int x0c = clamp(x0,     0, Nx-1);
        int x1c = clamp(x0 + 1, 0, Nx-1);
        int y0c = clamp(y0,     0, Ny-1);
        int y1c = clamp(y0 + 1, 0, Ny-1);

        float v00 = field[x0c * Ny + y0c];
        float v10 = field[x1c * Ny + y0c];
        float v01 = field[x0c * Ny + y1c];
        float v11 = field[x1c * Ny + y1c];

        return (1.0f-sx)*(1.0f-sy)*v00 + sx*(1.0f-sy)*v10
             + (1.0f-sx)*sy*v01 + sx*sy*v11;
    }

    // Generate reference trajectory via gradient ascent on info field
    // Returns flat array: [x0,y0,z0, x1,y1,z1, ...]
    std::vector<float> field_gradient_trajectory(
        const float* field, int Nx, int Ny,
        float field_ox, float field_oy, float field_res,
        float start_x, float start_y,
        int horizon, float ref_speed, float dt, float altitude
    ) const {
        std::vector<float> traj(horizon * 3);
        float cx = start_x, cy = start_y;
        float eps = field_res * 0.5f;

        for (int i = 0; i < horizon; ++i) {
            traj[i*3 + 0] = cx;
            traj[i*3 + 1] = cy;
            traj[i*3 + 2] = altitude;

            // Finite-difference gradient
            float vxp = interp2d(field, Nx, Ny, field_ox, field_oy, field_res, cx + eps, cy);
            float vxn = interp2d(field, Nx, Ny, field_ox, field_oy, field_res, cx - eps, cy);
            float vyp = interp2d(field, Nx, Ny, field_ox, field_oy, field_res, cx, cy + eps);
            float vyn = interp2d(field, Nx, Ny, field_ox, field_oy, field_res, cx, cy - eps);

            float gx = (vxp - vxn) / (2.0f * eps);
            float gy = (vyp - vyn) / (2.0f * eps);
            float gn = sqrtf(gx*gx + gy*gy) + 1e-8f;

            // Step along gradient at ref_speed
            float step = ref_speed * dt;
            cx += step * gx / gn;
            cy += step * gy / gn;
        }
        return traj;
    }
};

} // namespace planning
} // namespace mppi

#endif // MPPI_TRAJECTORY_GENERATOR_HPP
