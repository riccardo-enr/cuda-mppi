/**
 * @file map.cuh
 * @brief GPU-compatible occupancy grid structures (2D and 3D).
 *
 * Provides POD (Plain Old Data) structs — trivially copyable types with
 * no virtual functions or non-trivial constructors — that wrap a device
 * pointer to probability data and the grid metadata (dimensions,
 * resolution, origin). Being POD allows them to be passed by value to
 * CUDA kernels. Both structs support `__device__` and `__host__`
 * coordinate transforms.
 */

#ifndef MPPI_MAP_CUH
#define MPPI_MAP_CUH

#include <cuda_runtime.h>

#include <iostream>

namespace mppi {

/**
 * @brief 3D voxel occupancy grid.
 *
 * Stores occupancy probabilities $p \in [0, 1]$ in a flat array indexed
 * as $(z, y, x)$ in row-major order: `index = z * (W * H) + y * W + x`.
 */
struct OccupancyGrid {
  float* data;       ///< Occupancy probabilities $[W \times H \times D]$.
  int3 dims;         ///< Grid dimensions $(W, H, D)$ in cells.
  float resolution;  ///< Metres per cell.
  float3 origin;     ///< World coordinates of voxel $(0, 0, 0)$.

  /**
   * @brief Linearize a 3D cell index.
   * @return Flat index, or $-1$ if out of bounds.
   */
  __device__ __host__ int get_index(int x, int y, int z) const {
    if (x < 0 || x >= dims.x || y < 0 || y >= dims.y || z < 0 || z >= dims.z) {
      return -1;
    }
    return z * (dims.x * dims.y) + y * dims.x + x;
  }

  /**
   * @brief Convert world coordinates to grid cell indices.
   * @param pos  World-frame position.
   * @return     Integer cell indices (may be out of bounds).
   */
  __device__ __host__ int3 world_to_grid(float3 pos) const {
    int x = (int)floorf((pos.x - origin.x) / resolution);
    int y = (int)floorf((pos.y - origin.y) / resolution);
    int z = (int)floorf((pos.z - origin.z) / resolution);
    return make_int3(x, y, z);
  }

  /**
   * @brief Convert grid cell indices to world coordinates (voxel centre).
   * @param idx  Integer cell indices.
   * @return     World-frame position at the centre of the voxel.
   */
  __device__ __host__ float3 grid_to_world(int3 idx) const {
    float x = origin.x + (idx.x + 0.5f) * resolution;
    float y = origin.y + (idx.y + 0.5f) * resolution;
    float z = origin.z + (idx.z + 0.5f) * resolution;
    return make_float3(x, y, z);
  }

  /**
   * @brief Query occupancy probability at a world position.
   * @param pos  World-frame position.
   * @return     Occupancy probability, or $0.5$ if out of bounds (unknown).
   */
  __device__ float get_probability(float3 pos) const {
    int3 idx = world_to_grid(pos);
    int linear_idx = get_index(idx.x, idx.y, idx.z);
    if (linear_idx < 0) {
      return 0.5f;
    }
    return data[linear_idx];
  }

  /**
   * @brief Query occupancy probability by cell index.
   * @param idx  Integer cell indices.
   * @return     Occupancy probability, or $0.5$ if out of bounds.
   */
  __device__ float get_probability_idx(int3 idx) const {
    int linear_idx = get_index(idx.x, idx.y, idx.z);
    if (linear_idx < 0) {
      return 0.5f;
    }
    return data[linear_idx];
  }
};

/**
 * @brief 2D planar occupancy grid (for I-MPPI exploration).
 *
 * Stores occupancy probabilities $p \in [0, 1]$ in row-major order:
 * `index = y * W + x`.
 */
struct OccupancyGrid2D {
  float* data;       ///< Occupancy probabilities $[W \times H]$.
  int2 dims;         ///< Grid dimensions $(W, H)$ in cells.
  float resolution;  ///< Metres per cell.
  float2 origin;     ///< World coordinates of cell $(0, 0)$.

  /**
   * @brief Linearize a 2D cell index.
   * @return Flat index, or $-1$ if out of bounds.
   */
  __device__ __host__ int get_index(int x, int y) const {
    if (x < 0 || x >= dims.x || y < 0 || y >= dims.y) {
      return -1;
    }
    return y * dims.x + x;
  }

  /**
   * @brief Convert world coordinates to grid cell indices.
   * @param pos  World-frame 2D position.
   * @return     Integer cell indices (may be out of bounds).
   */
  __device__ __host__ int2 world_to_grid(float2 pos) const {
    int x = (int)floorf((pos.x - origin.x) / resolution);
    int y = (int)floorf((pos.y - origin.y) / resolution);
    return make_int2(x, y);
  }

  /**
   * @brief Convert grid cell indices to world coordinates (cell centre).
   * @param idx  Integer cell indices.
   * @return     World-frame 2D position at the centre of the cell.
   */
  __device__ __host__ float2 grid_to_world(int2 idx) const {
    float x = origin.x + (idx.x + 0.5f) * resolution;
    float y = origin.y + (idx.y + 0.5f) * resolution;
    return make_float2(x, y);
  }

  /**
   * @brief Query occupancy probability at a world position.
   * @param pos  World-frame 2D position.
   * @return     Occupancy probability, or $0.5$ if out of bounds (unknown).
   */
  __device__ float get_probability(float2 pos) const {
    int2 idx = world_to_grid(pos);
    int linear_idx = get_index(idx.x, idx.y);
    if (linear_idx < 0) {
      return 0.5f;
    }
    return data[linear_idx];
  }

  /**
   * @brief Query occupancy probability by cell index.
   * @param idx  Integer cell indices.
   * @return     Occupancy probability, or $0.5$ if out of bounds.
   */
  __device__ float get_probability_idx(int2 idx) const {
    int linear_idx = get_index(idx.x, idx.y);
    if (linear_idx < 0) {
      return 0.5f;
    }
    return data[linear_idx];
  }
};

}  // namespace mppi

#endif  // MPPI_MAP_CUH
