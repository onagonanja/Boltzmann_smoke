﻿#include "boltzmann_solver.hpp"
#include "vdb_exporter.hpp"
#include "visualizer.hpp"
#include "simulation_recorder.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip> 
#include <filesystem>
#include <nlohmann/json.hpp>

int main()
{
    try {
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

            // Load temperature boundary condition type
            if (j.contains("temperature_bc_type")) {
                std::string bc_type = j.value("temperature_bc_type", "adiabatic");
                if (bc_type == "adiabatic") {
                    init_params.temperature_bc_type = BoltzmannSolver::InitParams::TemperatureBCType::Adiabatic;
                } else if (bc_type == "dirichlet") {
                    init_params.temperature_bc_type = BoltzmannSolver::InitParams::TemperatureBCType::Dirichlet;
                } else if (bc_type == "periodic") {
                    init_params.temperature_bc_type = BoltzmannSolver::InitParams::TemperatureBCType::Periodic;
                } else {
                    init_params.temperature_bc_type = BoltzmannSolver::InitParams::TemperatureBCType::Adiabatic;
                }
            }
            init_params.dirichlet_temperature = j.value("dirichlet_temperature", 300.0f);

            // カメラ位置の読み込み
            if (j.contains("camera_pos") && j["camera_pos"].is_array() && j["camera_pos"].size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    init_params.camera_pos[i] = j["camera_pos"][i];
                }
            }

            // グリッドスケールの読み込み
            init_params.n_scale = j.value("n_scale", 1);
        }

        // Grid size settings based on n_scale
        const int nx = 32 * init_params.n_scale;
        const int ny = 32 * init_params.n_scale * 4;
        const int nz = 32 * init_params.n_scale;
        const int maxSteps = 300;
        const float dt = 0.1f;
        
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
            Visualizer visualizer(800, 600, init_params.camera_pos);
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
                if (saveSimulation && step % 1 == 0) {
                    std::string frameFilename = outputDir + "/frame_" + std::string(4 - std::to_string(step).length(), '0') + std::to_string(step) + ".vdb";
                    exporter.exportToVDB(frameFilename.c_str(), solver.getDensityData(), solver.getVelocityData());
                }
                
                // Wait to maintain 60FPS
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
