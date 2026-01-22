#!/bin/bash

# This script automates the process of running tests for the C++ solver project.

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Create a build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake to configure the project
cmake ..

# Build the project
make

# Run the tests
ctest --output-on-failure