#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <memory>

class SimulationRecorder {
public:
    SimulationRecorder(int nx, int ny, int nz);
    ~SimulationRecorder();

    // Save simulation results
    void saveFrame(const float* density);
    
    // Write saved results to file
    bool saveToFile(const std::string& filename);
    
    // Load saved results from file
    bool loadFromFile(const std::string& filename);
    
    // Get number of frames
    int getFrameCount() const { return frames_.size(); }
    
    // Get density data for specific frame
    const std::vector<float>& getFrame(int frame) const { return frames_[frame]; }
    
    // Get grid size
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNz() const { return nz_; }

private:
    int nx_, ny_, nz_;
    std::vector<std::vector<float>> frames_;  // Density data for each frame
}; 