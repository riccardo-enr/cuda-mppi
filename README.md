# CUDA MPPI

CUDA-accelerated Model Predictive Path Integral (MPPI) control library.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://riccardo-enr.github.io/cuda-mppi/)

## Overview

This library provides high-performance CUDA implementations of MPPI and related sampling-based control algorithms:

- **MPPI**: Model Predictive Path Integral control
- **KMPPI**: Kernel MPPI with colored noise
- **SMPPI**: Smooth MPPI variant

## Features

- Pure CUDA implementation for maximum performance
- Python bindings via nanobind
- Multiple controller variants
- Optimized CUDA kernels for parallel sampling

## Building

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
│   └── utils/         # CUDA utility functions
├── src/               # Implementation files
└── bindings/         # Python bindings
```

## Related Projects

- [jax_mppi](https://github.com/riccardo-enr/jax_mppi) — JAX-based MPPI implementation with Python bindings

## License

MIT License
