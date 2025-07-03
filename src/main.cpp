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
#include <nlohmann/json.hpp>

int main()
{
    try {
        // Grid size settings
        const int n_scale = 1;
        const int nx = 64 * n_scale;
        const int ny = 64 * n_scale * 2;
        const int nz = 64 * n_scale;
        const int maxSteps = 300;
        const float dt = 0.1f;
        
        // Whether to save simulation results
        bool saveSimulation = true;
        std::string saveFilename = "simulation_data.vdb";

        // Whether to replay simulation results
        bool replaySimulation = false;
        std::string replayFilename = "simulation_data.bin";

        // Load initialization parameters from JSON
        BoltzmannSolver::InitParams init_params;
        std::ifstream ifs("../init_params.json");
        if (ifs) {
            nlohmann::json j;
            ifs >> j;
            init_params.tau_f = j.value("tau_f", 1.3f);
            init_params.tau_t = j.value("tau_t", 0.8f);
            init_params.temperature = j.value("temperature", 300.0f);
            init_params.beta = j.value("beta", 0.1f);
            init_params.buoyancy_rand_ratio = j.value("buoyancy_rand_ratio", 0.8f);
            init_params.tau_rand_factor = j.value("tau_rand_factor", 0.2f);
            init_params.source_injection_rate = j.value("source_injection_rate", 0.1f);
            init_params.source_temperature = j.value("source_temperature", 500.0f);
            init_params.continuous_source = j.value("continuous_source", true);
            init_params.source_injection_interval = j.value("source_injection_interval", 10);
            init_params.simulation_steps_per_frame = j.value("simulation_steps_per_frame", 30);
            init_params.show_temperature_field = j.value("show_temperature_field", true);
            
            // Load wind parameters
            if (j.contains("wind")) {
                init_params.wind_base = j["wind"].value("base_wind", 0.05f);
                init_params.wind_factor = j["wind"].value("wind_factor", 0.2f);
            }
            
            // Load smoke source parameters
            if (j.contains("smoke_source")) {
                init_params.source_radius = j["smoke_source"].value("radius", 6.0f);
                init_params.source_density = j["smoke_source"].value("density", 0.5f);
            }
            
            // Load velocity limit parameter
            init_params.velocity_limit = j.value("velocity_limit", 0.3f);
        }

        if (replaySimulation) {
            // Replay simulation results
            SimulationRecorder recorder(nx, ny, nz);
            if (!recorder.loadFromFile(replayFilename)) {
                std::cerr << "Failed to load simulation results" << std::endl;
                return 1;
            }

            Visualizer visualizer(800, 600);
            int currentFrame = 0;
            
            // Create dummy temperature data for replay (since recorder only stores density)
            std::vector<float> dummy_temperature(nx * ny * nz, 300.0f);
            
            while (!visualizer.shouldClose()) {
                const auto& frame = recorder.getFrame(currentFrame);
                visualizer.update(frame.data(), dummy_temperature.data(), nx, ny, nz, init_params.show_temperature_field);
                currentFrame = (currentFrame + 1) % recorder.getFrameCount();
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
        } else {
            // Normal simulation execution
            BoltzmannSolver solver(nx, ny, nz, init_params);
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
                visualizer.update(solver.getDensityData(), solver.getTemperatureData(), nx, ny, nz, init_params.show_temperature_field);
                
                // Update simulation
                solver.simulate(dt, init_params.simulation_steps_per_frame);
                
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
