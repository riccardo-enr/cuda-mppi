#ifndef FSMI_COST_CUH
#define FSMI_COST_CUH

#include <cuda_runtime.h>
#include "mppi/core/mppi_common.cuh"
#include "mppi/core/map.cuh"

namespace mppi {

struct FSMICost {
    const OccupancyGrid* map; // Device pointer to map struct
    float lambda_info;        // Information gain weight
    float sensor_range;       // Max range in meters
    
    // Additional parameters for dynamics/collision can be added.
    
    __device__ float compute(const float* x, const float* u, int t) const {
        // 1. Motion Cost (Regularization)
        float cost = 0.0f;
        // Simple quadratic control cost: u^T R u
        // Assuming nu=2 (v, w)
        cost += 0.1f * (u[0]*u[0] + u[1]*u[1]);

        // 2. Information Reward (FSMI)
        if (map == nullptr) return cost;

        // Robot State: Assuming standard 2D/3D state vector
        // [x, y, z, roll, pitch, yaw, ...] or [x, y, theta, v, w]
        // Let's assume indices: 0:x, 1:y, 2:z (or theta if 2D).
        // For the 2D test dynamics: x, y, vx, vy. No theta? 
        // The test dynamics don't have orientation. 
        // Let's assume the robot "looks" in the direction of velocity 
        // OR we need to augment state with yaw.
        // For MPPI, usually we optimize (v, w) so we track yaw.
        // Let's assume state index 2 is yaw? 
        // In the test setup: x, y, vx, vy.
        // Let's calculate yaw from velocity for this test: atan2(vy, vx).
        
        float rx = x[0];
        float ry = x[1];
        float rz = 0.0f; // Fixed height (0 for 2D map)
        
        float vx = x[2];
        float vy = x[3];
        float yaw = atan2f(vy, vx);
        
        // Raycast parameters
        float max_dist = (sensor_range > 0.0f) ? sensor_range : 10.0f;
        float step_size = map->resolution; // Step size = grid resolution
        int num_steps = (int)(max_dist / step_size);
        
        float current_vis = 1.0f; // P(visible)
        float total_info = 0.0f;
        
        float cx = rx;
        float cy = ry;
        float cz = rz;
        
        float dx = cosf(yaw) * step_size;
        float dy = sinf(yaw) * step_size;
        
        // Simple Raymarching (0-order)
        // A better approach is Bresenham or DDA, but for GPU float evaluation,
        // stepping by resolution is often "good enough" for probabilistic maps.
        
        for(int k=0; k<num_steps; ++k) {
            cx += dx;
            cy += dy;
            
            // Get Probability of Occupancy
            // Use 3D accessor
            float p = map->get_probability(make_float3(cx, cy, cz));
            
            // Clamp p to avoid log(0)
            if(p < 0.001f) p = 0.001f;
            if(p > 0.999f) p = 0.999f;
            
            // Shannon Entropy: H(p) = -p log p - (1-p) log (1-p)
            // Max at p=0.5 -> H = -0.5*-0.693... * 2 = 0.693 nats
            float entropy = -p * logf(p) - (1.0f - p) * logf(1.0f - p);
            
            // Expected Information Gain at this cell
            // We gain info if we see it.
            // IG += P(visible) * Entropy
            total_info += current_vis * entropy;
            
            // Update visibility
            // P(visible_next) = P(visible_current) * P(empty)
            // P(empty) = 1 - p
            current_vis *= (1.0f - p);
            
            if (current_vis < 0.01f) break; // Optimization: stop if visibility is low
        }
        
        // Subtract Info Reward from Cost (Maximizing Info = Minimizing Cost)
        cost -= lambda_info * total_info;
        
        return cost;
    }
    
    __device__ float terminal_cost(const float* x) const {
        return 0.0f;
    }
};

} // namespace mppi

#endif // FSMI_COST_CUH
