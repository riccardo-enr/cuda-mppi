# Justfile

# Default recipe
default:
    @just --list

# Create virtual environment and install dependencies
setup:
    uv venv .venv --python 3.12
    uv pip install -e .[dev]

# Configure CMake build
configure:
    mkdir -p build
    cd build && cmake .. -Dnanobind_DIR=$(uv run python -m nanobind --cmake_dir) -DPython_EXECUTABLE=$(which python3)

# Build all C++ targets
build: configure
    cd build && make -j$(nproc)

# Build only tests
build-tests: configure
    cd build && make mppi_gtest pendulum_test i_mppi_test fsmi_unit_test

# Build only examples
build-examples: configure
    cd build && make i_mppi_sim informative_sim mppi_log_trajectories

# Run GTest suite
test-gtest: build-tests
    ./build/tests/mppi_gtest

# Run pendulum test
test-pendulum: build-tests
    ./build/tests/pendulum_test

# Run FSMI unit test
test-fsmi: build-tests
    ./build/tests/fsmi_unit_test

# Run all C++ tests
test-cpp: build-tests
    ./build/tests/mppi_gtest
    ./build/tests/pendulum_test
    ./build/tests/fsmi_unit_test
    ./build/tests/i_mppi_test

# Run I-MPPI simulation
run-i-mppi: build-examples
    ./build/examples/i_mppi_sim

# Run informative simulation (quadrotor + FSMI + info field)
run-informative-sim: build-examples
    ./build/examples/informative_sim

# Log trajectories to CSV and plot
log-trajectories: build-examples
    ./build/examples/mppi_log_trajectories
    uv run scripts/plot_mppi_tests.py

# Analyze exploration behavior
analyze-exploration: run-i-mppi
    uv run scripts/analyze_exploration.py

analyze-exploration-python:
    uv run scripts/analyze_exploration.py

# Clean build artifacts
clean:
    rm -rf build
    rm -rf dist
    rm -rf *.egg-info
    rm -rf .venv

# Format code (simulated for C++, actual for Python if added)
format:
    uv run ruff format .

# Run python tests
test-py:
    PYTHONPATH= uv run pytest tests

# Run all tests
test: test-cpp test-py

# Documentation
quarto-doc:
    uv run quarto preview docs

# Publish documentation to GitHub Pages manually
publish-doc:
    cd docs && quarto publish gh-pages --no-browser --no-prompt
