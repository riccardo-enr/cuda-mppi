# Plan: I-MPPI (Informative MPPI) Development

This plan outlines the steps for developing the `i_mppi` controller variant in the `cuda-mppi` library, based on the "Hierarchical Informative MPPI" framework.

## Goal
Implement `IMPPIController`, a controller that integrates:
1.  **Biased Sampling**: A mixture distribution of samples centered around the previous solution and a global reference trajectory.
2.  **Information Cost**: A cost term based on Fast Shannon Mutual Information (FSMI) to reward exploration.

## Steps

### 1. Research and Definition
- [x] Analyze `/home/riccardo/phd/notes/delft/i_mppi_note.qmd`.
- [x] Define strategy:
    -   **Biased Sampling**: Implemented by shifting the mean of the noise buffer for a subset of samples. $u = u_{nom} + \epsilon$, where $\epsilon$ is shifted such that effectively $u \sim \mathcal{N}(u_{ref}, \Sigma)$.
    -   **FSMI Cost**: A custom `Cost` functor that performs raycasting on a GPU-resident map (Occupancy Grid for MVP).

### 2. Infrastructure Setup
- [x] Create development branch `feat/i-mppi`.
- [x] Create development folder `i_mppi/`.
- [ ] Initialize `include/mppi/controllers/i_mppi.cuh`.

### 3. Implementation
- [x] **Config**: Add `alpha` (bias ratio), `lambda_info` (info gain weight) to `MPPIConfig` (or a subclass).
- [x] **Map Representation**: Create a simple `OccupancyGrid` CUDA struct in `include/mppi/core/map.cuh` (new file) to support raycasting.
- [x] **FSMI Cost**: Implement `FSMICost` in `include/mppi/instantiations/fsmi_cost.cuh`.
    -   Implement `raycast` and `compute_info_gain` device functions.
- [x] **Controller**: Implement `IMPPIController` in `include/mppi/controllers/i_mppi.cuh`.
    -   Manage `u_ref` (reference trajectory).
    -   Implement `shift_noise_means` kernel/function to apply bias.

### 4. Testing & Verification
- [x] Create a standalone test in `src/i_mppi_test.cu`.
    -   Setup a mock map (e.g., a wall with a hole or a simple room).
    -   Verify that the controller is attracted to unknown areas.
- [x] Benchmark performance of the raycasting kernel. (Implicitly verified functionality via test run)
- [x] Conduct extensive testing campaign and generate documentation images (`src/i_mppi_sim.cu`).

### 5. Integration
- [ ] Expose new classes to Python bindings if needed (low priority for CLI task, focus on C++ core).
- [x] Cleanup and Documentation update.

## Progress Tracking
- 2026-02-05: Updated plan with specific I-MPPI details.
- 2026-02-05: Implemented IMPPIController, FSMICost, OccupancyGrid and verified with i_mppi_test.
- 2026-02-05: Conducted extensive testing campaign, generated comparison plots and updated documentation.
