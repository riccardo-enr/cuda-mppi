# CLAUDE.md

## Build

```bash
pixi run cmake-configure   # configure with pixi
pixi run cmake-build        # build all targets
```

Or manually:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Issue-closing checklist

Before closing any GitHub issue, run the relevant tests below and paste the output in the closing comment.

### Trajectory tracking controllers (MPPI, I-MPPI, S-MPPI, K-MPPI, BSpline-MPPI)

```bash
pixi run python tests/test_quadrotor_tracking.py
```

- Total RMSE must be <= 0.70 m (paper baseline)
- Test must print "Tracking test passed"

### Core kernel / config changes

```bash
./build/tests/i_mppi_gtest
./build/tests/bspline_test
./build/tests/fsmi_unit_test
```

All tests must report `[  PASSED  ]`.

### FSMI / occupancy grid changes

```bash
./build/tests/fsmi_unit_test
```

### Python bindings changes

```bash
pixi run pytest tests/ -v
```

### Closing comment format

Include in the closing comment:
- Which tests were run
- RMSE value (for tracking controllers)
- Number of tests passed / total