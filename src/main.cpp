#include "boltzmann_solver.hpp"
#include "vdb_exporter.hpp"
#include "visualizer.hpp"
#include "simulation_recorder.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>  // Added for std::setw, std::setfill
#include <filesystem>  // For filesystem operations

int main()
{
    try {
        // Grid size settings
        const int n_scale = 1;
        const int nx = 128 * n_scale;
        const int ny = 128 * n_scale * 2;
        const int nz = 128 * n_scale;
        const int maxSteps = 100;
        const float dt = 0.1f;
        const float viscosity = 0.1f;
        const float diffusion = 0.1f;
        
        // Whether to save simulation results
        bool saveSimulation = true;
        std::string saveFilename = "simulation_data.vdb";

        // Whether to replay simulation results
        bool replaySimulation = false;
        std::string replayFilename = "simulation_data.bin";

        if (replaySimulation) {
            // Replay simulation results
            SimulationRecorder recorder(nx, ny, nz);
            if (!recorder.loadFromFile(replayFilename)) {
                std::cerr << "Failed to load simulation results" << std::endl;
                return 1;
            }

            Visualizer visualizer(800, 600);
            int currentFrame = 0;
            while (!visualizer.shouldClose()) {
                const auto& frame = recorder.getFrame(currentFrame);
                visualizer.update(frame.data(), nx, ny, nz);
                currentFrame = (currentFrame + 1) % recorder.getFrameCount();
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
        } else {
            // Normal simulation execution
            BoltzmannSolver solver(nx, ny, nz);
            Visualizer visualizer(800, 600);  // Create 800x600 window
            VDBExporter exporter(nx, ny, nz);
            
            // Initial state setup
            solver.initialize();
            
            // Create folder for VDB file storage
            std::string outputDir = "vdb_output";
            if (!std::filesystem::exists(outputDir)) {
                std::filesystem::create_directory(outputDir);
            }
            
            for (int step = 0; step < maxSteps && !visualizer.shouldClose(); ++step) {
                // Update visualizer
                visualizer.update(solver.getDensityData(), nx, ny, nz);
                
                // Update simulation
                solver.simulate(dt, 30);
                
                // Output to OpenVDB file
                if (saveSimulation && step % 1 == 0) {  // Save every 1 step
                    std::string frameFilename = outputDir + "/frame_" + std::string(4 - std::to_string(step).length(), '0') + std::to_string(step) + ".vdb";
                    exporter.exportToVDB(frameFilename.c_str(), solver.getDensityData(), solver.getVelocityData());
                }
                
                // Wait to maintain 60FPS
                std::this_thread::sleep_for(std::chrono::milliseconds(16)); // 16ms -> 160ms
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
