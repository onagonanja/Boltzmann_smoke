#include "simulation_recorder.hpp"
#include <iostream>

SimulationRecorder::SimulationRecorder(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz) {
}

SimulationRecorder::~SimulationRecorder() {
}

void SimulationRecorder::saveFrame(const float* density) {
    std::vector<float> frame(nx_ * ny_ * nz_);
    std::copy(density, density + nx_ * ny_ * nz_, frame.begin());
    frames_.push_back(std::move(frame));
}

bool SimulationRecorder::saveToFile(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Write header information
    file.write(reinterpret_cast<const char*>(&nx_), sizeof(nx_));
    file.write(reinterpret_cast<const char*>(&ny_), sizeof(ny_));
    file.write(reinterpret_cast<const char*>(&nz_), sizeof(nz_));
    int frameCount = frames_.size();
    file.write(reinterpret_cast<const char*>(&frameCount), sizeof(frameCount));

    // Write data for each frame
    for (const auto& frame : frames_) {
        file.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(float));
    }

    return true;
}

bool SimulationRecorder::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Read header information
    file.read(reinterpret_cast<char*>(&nx_), sizeof(nx_));
    file.read(reinterpret_cast<char*>(&ny_), sizeof(ny_));
    file.read(reinterpret_cast<char*>(&nz_), sizeof(nz_));
    int frameCount;
    file.read(reinterpret_cast<char*>(&frameCount), sizeof(frameCount));

    // Read frame data
    frames_.clear();
    frames_.reserve(frameCount);
    for (int i = 0; i < frameCount; ++i) {
        std::vector<float> frame(nx_ * ny_ * nz_);
        file.read(reinterpret_cast<char*>(frame.data()), frame.size() * sizeof(float));
        frames_.push_back(std::move(frame));
    }

    return true;
} 