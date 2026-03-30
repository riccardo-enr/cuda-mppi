# GTest-based MPPI tests
add_executable(mppi_gtest mppi_gtest.cu)
target_link_libraries(mppi_gtest PRIVATE cuda_mppi_lib GTest::gtest_main)
target_include_directories(mppi_gtest PRIVATE ${UAV_CONTROL_INCLUDE_DIR})
add_test(NAME mppi_gtest COMMAND mppi_gtest)

# I-MPPI per-component GTest suite
add_executable(i_mppi_gtest i_mppi_gtest.cu)
target_link_libraries(i_mppi_gtest PRIVATE cuda_mppi_lib GTest::gtest_main)
add_test(NAME i_mppi_gtest COMMAND i_mppi_gtest)

# Standalone test executables
add_executable(pendulum_test pendulum_test.cu)
target_link_libraries(pendulum_test PRIVATE cuda_mppi_lib)

add_executable(i_mppi_test i_mppi_test.cu)
target_link_libraries(i_mppi_test PRIVATE cuda_mppi_lib)

add_executable(fsmi_unit_test fsmi_unit_test.cu)
target_link_libraries(fsmi_unit_test PRIVATE cuda_mppi_lib)

add_executable(bspline_test bspline_test.cu)
target_link_libraries(bspline_test PRIVATE cuda_mppi_lib)
