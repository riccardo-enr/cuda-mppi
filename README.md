# CUDA MPPI

CUDA-accelerated Model Predictive Path Integral (MPPI) control library.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://riccardo-enr.github.io/cuda-mppi/)

## Overview

This library provides high-performance CUDA implementations of MPPI and related sampling-based control algorithms:

- **MPPI**: Model Predictive Path Integral control
- **KMPPI**: Kernel MPPI with colored noise
- **SMPPI**: Smooth MPPI variant
- **JIT MPPI**: Just-In-Time compiled MPPI controllers

## Features

- Pure CUDA implementation for maximum performance
- Python bindings via nanobind
- JIT compilation support for custom dynamics
- Multiple controller variants
- Optimized CUDA kernels for parallel sampling

## Building

This library is designed to be built as part of the [jax_mppi](https://github.com/riccardo-enr/jax_mppi) project using CMake and scikit-build-core.

### Standalone Build (Optional)

```bash
mkdir build && cd build
cmake ..
make
```

### Requirements

- CUDA Toolkit (>= 11.0)
- CMake (>= 3.18)
- C++17 compiler
- nanobind (for Python bindings)

## Structure

```text
cuda_mppi/
├── include/mppi/
│   ├── core/           # Core CUDA kernels and utilities
│   ├── controllers/    # MPPI controller implementations
│   ├── instantiations/ # Example dynamics models
│   ├── jit/           # JIT compilation support
│   └── utils/         # CUDA utility functions
├── src/               # Implementation files
│   └── jit/          # JIT compiler implementation
└── bindings/         # Python bindings
```

## License

MIT License - see the main [jax_mppi repository](https://github.com/riccardo-enr/jax_mppi) for details.

## Usage

This library is primarily used through the jax_mppi Python package. See the main repository for examples and documentation.
