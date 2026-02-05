# Justfile

# Default recipe
default:
    @just --list

# Create virtual environment and install dependencies
setup:
    uv venv .venv --python 3.12
    uv pip install -e .[dev]

# Build the C++ tests
build-cpp:
    mkdir -p build
    cd build && cmake .. -Dnanobind_DIR=$(uv run python -m nanobind --cmake_dir) -DPython_EXECUTABLE=$(which python3) && make pendulum_test

# Run the C++ pendulum test
test-cpp: build-cpp
    ./build/pendulum_test

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
