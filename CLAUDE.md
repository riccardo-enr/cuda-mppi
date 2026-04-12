# CLAUDE.md

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Pixi tasks

```bash
pixi run install    # pip install the Python bindings
pixi run test       # pytest tests/ -v
pixi run tracking   # quadrotor tracking test with plot
```

## Issue-closing checklist

Before closing any GitHub issue, run **all applicable** test categories below (union of all that apply) and paste the output in the closing comment.

For changes that span multiple categories, run all relevant suites.
For new features, add tests or justify in the closing comment why none were added.

### Trajectory tracking controllers (MPPI, I-MPPI, S-MPPI, K-MPPI, BSpline-MPPI)

```bash
pixi run tracking
```

- The automated test asserts RMSE < 2.0 m (regression guard)
- Paper baseline is 0.69 m; report the actual RMSE in the closing comment
- If RMSE regresses significantly above ~0.70 m, investigate before closing

### Core kernel / config changes

```bash
./build/tests/mppi_gtest
./build/tests/i_mppi_gtest
./build/tests/pendulum_test
./build/tests/bspline_test
./build/tests/fsmi_unit_test
```

GTest suites (`mppi_gtest`, `i_mppi_gtest`) must report `[  PASSED  ]`.
Other binaries (`pendulum_test`, `bspline_test`, `fsmi_unit_test`) print `PASSED` on success.

### FSMI / occupancy grid changes

```bash
./build/tests/fsmi_unit_test
```

### Python bindings changes

```bash
pixi run install
pixi run test
```

### Closing comment format

Include in the closing comment:
- Which tests were run
- RMSE value (for tracking controllers)
- Number of tests passed / total
