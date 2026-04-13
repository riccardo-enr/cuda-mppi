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

## JAX vs CUDA MPPI

Apples-to-apples comparison on a quadrotor lemniscate tracking task (K=900,
N=40, dt=0.02 s, 30 s simulation). Both implementations run the same MPPI
algorithm on the same GPU -- only the stack differs.

### Trajectory tracking

![3D trajectory comparison](docs/_media/compare_trajectory_3d.png)

### Compute latency

![Compute latency](docs/_media/compare_latency.png)

Reproduce with:

```bash
pixi run python scripts/bench_jax_mppi.py --compare --plot
```

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
