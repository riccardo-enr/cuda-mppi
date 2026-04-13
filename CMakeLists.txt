cmake_minimum_required(VERSION 3.18)
project(cuda_mppi LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Suppress Eigen/CUDA constexpr warnings
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module QUIET)
find_package(nanobind CONFIG QUIET)

enable_testing()

# JIT compiler sources (compiled into the Python extension)
set(JIT_SOURCES
    src/jit/jit_compiler.cpp
    src/jit/jit_mppi_controller.cpp
)

# Header-only library
add_library(cuda_mppi_lib INTERFACE)

target_link_libraries(cuda_mppi_lib INTERFACE
    Eigen3::Eigen
    CUDA::curand
    CUDA::nvrtc
    CUDA::cuda_driver
)
target_include_directories(cuda_mppi_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

# Python extension
if(nanobind_FOUND AND Python_FOUND)
  nanobind_add_module(cuda_mppi bindings/bindings.cu ${JIT_SOURCES})
  target_link_libraries(cuda_mppi PRIVATE cuda_mppi_lib)
  install(TARGETS cuda_mppi DESTINATION .)
endif()

install(DIRECTORY include/mppi DESTINATION include)

# --- CMake package config for find_package(cuda_mppi) ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS cuda_mppi_lib EXPORT cuda_mppiTargets)

configure_package_config_file(
    cmake/cuda_mppiConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_mppiConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_mppi
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cuda_mppiConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_mppi
)
install(EXPORT cuda_mppiTargets
    FILE cuda_mppiTargets.cmake
    NAMESPACE cuda_mppi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_mppi
)

# Cross-repo dependency (for tests/examples that use uav_control headers)
set(UAV_CONTROL_INCLUDE_DIR ""
    CACHE PATH "Path to uav_control headers (unused — AccelerationTrackingCost now lives in-repo)")

# Build options — disabled under scikit-build-core (pip install)
if(DEFINED SKBUILD)
  option(CUDA_MPPI_BUILD_TESTS "Build tests" OFF)
  option(CUDA_MPPI_BUILD_EXAMPLES "Build examples" OFF)
else()
  option(CUDA_MPPI_BUILD_TESTS "Build tests" ON)
  option(CUDA_MPPI_BUILD_EXAMPLES "Build examples" ON)
endif()

if(CUDA_MPPI_BUILD_TESTS)
  find_package(GTest REQUIRED)
  add_subdirectory(tests)
endif()

if(CUDA_MPPI_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
