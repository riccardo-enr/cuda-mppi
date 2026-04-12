## Summary

<!-- Brief description of what this PR does -->

## Test results

<!-- Run the relevant tests from the checklist below and paste output -->

### Tracking controllers (if applicable)
- [ ] `pixi run python tests/test_quadrotor_tracking.py`
- RMSE: <!-- e.g. 0.596 m -->

### Core kernel / config (if applicable)
- [ ] `./build/tests/i_mppi_gtest`
- [ ] `./build/tests/bspline_test`
- [ ] `./build/tests/fsmi_unit_test`

### FSMI / occupancy (if applicable)
- [ ] `./build/tests/fsmi_unit_test`

### Python bindings (if applicable)
- [ ] `pixi run pytest tests/ -v`

## Closes

<!-- e.g. closes #19 -->
