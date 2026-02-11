#ifndef MPPI_MAP_CUH
#define MPPI_MAP_CUH

#include <cuda_runtime.h>
#include <iostream>

namespace mppi {

struct OccupancyGrid {
    float* data;          // Probability [0, 1] (size: width * height * depth)
    int3 dims;            // Dimensions (x, y, z) in number of cells
    float resolution;     // Meters per cell
    float3 origin;        // World coordinates of index (0,0,0)

    // Helper to linearize index
    __device__ __host__ int get_index(int x, int y, int z) const {
        if (x < 0 || x >= dims.x || y < 0 || y >= dims.y || z < 0 || z >= dims.z) return -1;
        return z * (dims.x * dims.y) + y * dims.x + x;
    }

    // World to Grid
    __device__ __host__ int3 world_to_grid(float3 pos) const {
        int x = (int)((pos.x - origin.x) / resolution);
        int y = (int)((pos.y - origin.y) / resolution);
        int z = (int)((pos.z - origin.z) / resolution);
        return make_int3(x, y, z);
    }
    
    // Grid to World (center of voxel)
    __device__ __host__ float3 grid_to_world(int3 idx) const {
        float x = origin.x + (idx.x + 0.5f) * resolution;
        float y = origin.y + (idx.y + 0.5f) * resolution;
        float z = origin.z + (idx.z + 0.5f) * resolution;
        return make_float3(x, y, z);
    }

    // Accessor
    __device__ float get_probability(float3 pos) const {
        int3 idx = world_to_grid(pos);
        int linear_idx = get_index(idx.x, idx.y, idx.z);
        if (linear_idx < 0) return 0.5f; // Unknown/outside
        return data[linear_idx];
    }
    
    __device__ float get_probability_idx(int3 idx) const {
        int linear_idx = get_index(idx.x, idx.y, idx.z);
        if (linear_idx < 0) return 0.5f;
        return data[linear_idx];
    }
};

// ---------------------------------------------------------------------------
// 2D Occupancy Grid (for I-MPPI planar exploration)
// ---------------------------------------------------------------------------
struct OccupancyGrid2D {
    float* data;          // Probability [0, 1] (size: width * height)
    int2 dims;            // (width, height) in cells
    float resolution;     // Meters per cell
    float2 origin;        // World coordinates of cell (0, 0)

    __device__ __host__ int get_index(int x, int y) const {
        if (x < 0 || x >= dims.x || y < 0 || y >= dims.y) return -1;
        return y * dims.x + x;
    }

    __device__ __host__ int2 world_to_grid(float2 pos) const {
        int x = (int)((pos.x - origin.x) / resolution);
        int y = (int)((pos.y - origin.y) / resolution);
        return make_int2(x, y);
    }

    __device__ __host__ float2 grid_to_world(int2 idx) const {
        float x = origin.x + (idx.x + 0.5f) * resolution;
        float y = origin.y + (idx.y + 0.5f) * resolution;
        return make_float2(x, y);
    }

    __device__ float get_probability(float2 pos) const {
        int2 idx = world_to_grid(pos);
        int linear_idx = get_index(idx.x, idx.y);
        if (linear_idx < 0) return 0.5f;  // Unknown/outside
        return data[linear_idx];
    }

    __device__ float get_probability_idx(int2 idx) const {
        int linear_idx = get_index(idx.x, idx.y);
        if (linear_idx < 0) return 0.5f;
        return data[linear_idx];
    }
};

} // namespace mppi

#endif // MPPI_MAP_CUH
