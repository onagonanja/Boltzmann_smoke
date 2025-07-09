# Boltzmann Smoke Simulation

> This project is currently under active development.

A GPU-accelerated smoke simulation using the Lattice Boltzmann Method (LBM) to generate realistic smoke behavior. The simulation produces smoke particles based on fluid dynamics principles and exports the results in OpenVDB format for external rendering and post-processing.

## Quick Build Guide

### Prerequisites

- CUDA Toolkit (version 11.0 or higher)
- CMake (version 3.18 or higher)
- C++ Compiler with C++17 support
- Required libraries: OpenVDB, TBB, Imath, GLEW, GLFW, nlohmann/json

### Windows Build

```bash
# Install dependencies using vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install
./vcpkg install openvdb:x64-windows tbb:x64-windows imath:x64-windows glew:x64-windows glfw3:x64-windows nlohmann-json:x64-windows

# Build the project
cd Boltzmann_smoke
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

### Linux Build

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install libopenvdb-dev libtbb-dev libimath-dev libglew-dev libglfw3-dev nlohmann-json3-dev

# Build the project
mkdir build && cd build
cmake ..
make
```

### Run the Simulation

```bash
./boltzmann_smoke
```

## Configuration

Edit `init_params.json` to adjust simulation parameters including camera position, temperature settings, and smoke source properties.
