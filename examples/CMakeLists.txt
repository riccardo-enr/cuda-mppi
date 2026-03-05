add_executable(i_mppi_sim i_mppi_sim.cu)
target_link_libraries(i_mppi_sim PRIVATE cuda_mppi_lib)

# Full informative simulation (quadrotor + FSMI + info field)
add_executable(informative_sim informative_sim.cu)
target_link_libraries(informative_sim PRIVATE cuda_mppi_lib)

# Trajectory logging for visualization
add_executable(mppi_log_trajectories mppi_log_trajectories.cu)
target_link_libraries(mppi_log_trajectories PRIVATE cuda_mppi_lib)
target_include_directories(mppi_log_trajectories PRIVATE ${UAV_CONTROL_INCLUDE_DIR})
target_compile_definitions(mppi_log_trajectories PRIVATE
    DATA_DIR="${CMAKE_SOURCE_DIR}/data")
