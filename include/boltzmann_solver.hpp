#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>

// Forward declaration
class VDBExporter;

class BoltzmannSolver
{
public:
    struct InitParams {
        float tau_f = 1.3f;
        float tau_t = 0.8f;
        float temperature = 300.0f;
        float wind_base = 0.05f;
        float wind_factor = 0.2f;
        float source_radius = 6.0f;
        float source_density = 0.5f;
        float velocity_limit = 0.3f;
        float beta = 0.1f;  // Thermal expansion coefficient
        float buoyancy_rand_ratio = 0.8f;  // Buoyancy force randomization ratio
        float tau_rand_factor = 0.2f;  // Relaxation time randomization factor
        // Add more parameters as needed
    };

    BoltzmannSolver(int nx, int ny, int nz, const InitParams& params);
    ~BoltzmannSolver();

    // Simulation initialization
    void initialize();

    // Simulation execution
    void simulate(float dt, int num_steps);

    // Output to OpenVDB file
    void exportToVDB(const char *filename, VDBExporter* exporter);

    // Data access functions
    const float* getDensityData() const { return h_density; }
    const float* getVelocityData() const { return h_vel_.data(); }
    const float* getTemperatureData() const { return h_temperature; }

    // Statistical functions
    int countActiveVoxels() const;
    float getMaxDensity() const;
    float getAverageDensity() const;
    float getMaxTemperature() const;
    float getAverageTemperature() const;
    float getAverageVelocity() const;
    float getMaxVelocity() const;

private:
    int nx_, ny_, nz_;  // Grid size
    int current_step_ = 0;  // Current simulation step count
    InitParams init_params_;

    // GPU memory - for fluid simulation
    float* d_f_distribution;  // Fluid distribution function
    float* d_density;      // Density
    float* d_velocity_x;   // X-direction velocity
    float* d_velocity_y;   // Y-direction velocity
    float* d_velocity_z;   // Z-direction velocity
    float* d_tau_f;       // Fluid relaxation time

    // GPU memory - for heat conduction simulation
    float* d_g_distribution;  // Temperature distribution function
    float* d_temperature;    // Temperature field
    float* d_tau_t;         // Temperature relaxation time

    // CPU memory
    float* h_density;     // Density
    float* h_temperature; // Temperature field
    std::vector<float> h_vel_;  // Velocity
    std::vector<float> h_rho_;  // Density
    std::vector<float> h_tau_f_;  // Fluid relaxation time
    std::vector<float> h_tau_t_;  // Temperature relaxation time

    // Helper functions
    void allocateMemory();
    void freeMemory();
    void initializeFields();
    
    // Fluid simulation functions
    void streamFluid();
    void collideFluid();
    void updateMacroscopic();
    
    // Heat conduction simulation functions
    void streamTemperature();
    void collideTemperature();
    void updateTemperature();
    
    void copyToHost();  // Data copy from GPU to CPU
};