﻿#ifdef _MSC_VER
extern "C" {
    __declspec(dllexport) void __cudaRegisterLinkedBinary_e68baee6_19_boltzmann_solver_cu_c_dx(void) {}
}
#endif

#include "boltzmann_solver.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <curand_kernel.h>

// D3Q19 model velocity vectors
__constant__ float c_dx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__constant__ float c_dy[19] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
__constant__ float c_dz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};

// Weight coefficients
__constant__ float w[19] = {
    1.0f/3.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

// D3Q7 model constants (for temperature field)
__device__ __constant__ float c_t_dx[7] = {0.0f,  1.0f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f};
__device__ __constant__ float c_t_dy[7] = {0.0f,  0.0f,  0.0f,  1.0f, -1.0f,  0.0f,  0.0f};
__device__ __constant__ float c_t_dz[7] = {0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  1.0f, -1.0f};
__device__ __constant__ float w_t[7] = {1.0f/4.0f, 1.0f/8.0f, 1.0f/8.0f, 1.0f/8.0f, 1.0f/8.0f, 1.0f/8.0f, 1.0f/8.0f};

__device__ int found_negative_f = 0;
__device__ int found_negative_g = 0;

__constant__ int focused_point_x = 64;
__constant__ int focused_point_y = 32;
__constant__ int focused_point_z = 49;


// Fluid collision step
__global__ void fluidCollisionKernel(float* f, float* rho, float* vel_x, float* vel_y, float* vel_z, float* tau_f, float* temperature, int nx, int ny, int nz, int current_step, float velocity_limit, float beta, float buoyancy_rand_ratio, float tau_rand_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    // Initialize random number generator
    curandState state;
    curand_init(clock64(), idx + current_step, 0, &state);

    float f_eq[19];
    float rho_local = rho[idx];
    float3 vel_local = make_float3(vel_x[idx], vel_y[idx], vel_z[idx]);

    // Calculate buoyancy force based on temperature
    const float g = 9.81f;
    const float T_ref = 300.0f;
    float T_local = temperature[idx];
    float rand_factor = buoyancy_rand_ratio + (1.0f - buoyancy_rand_ratio) * curand_uniform(&state);
    float buoyancy_force = beta * g * (T_local - T_ref) * rand_factor;
    vel_local.y += buoyancy_force;

    // Limit velocity
    float u_sq = vel_local.x * vel_local.x + vel_local.y * vel_local.y + vel_local.z * vel_local.z;
    float u_mag = sqrtf(u_sq);
    if (u_mag > velocity_limit) {
        float scale = velocity_limit / u_mag;
        vel_local.x *= scale;
        vel_local.y *= scale;
        vel_local.z *= scale;
        u_sq = velocity_limit * velocity_limit;
    }

    // Calculate equilibrium distribution
    for (int i = 0; i < 19; i++) {
        float ci_dot_u = c_dx[i] * vel_local.x + c_dy[i] * vel_local.y + c_dz[i] * vel_local.z;
        f_eq[i] = w[i] * rho_local * (1.0f + 3.0f * ci_dot_u + 4.5f * ci_dot_u * ci_dot_u - 1.5f * u_sq);
    }

    float tau = tau_f[idx] + tau_rand_factor * curand_uniform(&state);
    float c_s2 = 1.0f / 3.0f;
    float Fy = buoyancy_force; // y方向の外力

    for (int i = 0; i < 19; i++) {
        float ci_dot_u = c_dx[i] * vel_local.x + c_dy[i] * vel_local.y + c_dz[i] * vel_local.z;
        float Fi = w[i] * ( (c_dy[i] - vel_local.y) / c_s2 + ci_dot_u * c_dy[i] / (c_s2 * c_s2)) * Fy;
        f[19*idx + i] = f[19*idx + i] - (1.0f/tau) * (f[19*idx + i] - f_eq[i]) + 0.0 *(1.0f - 0.5f / tau) * Fi;
    }
}

// Temperature field collision step
__global__ void temperatureCollisionKernel(float* g, float* temperature, float* vel_x, float* vel_y, float* vel_z, float* tau_t, int nx, int ny, int nz, float velocity_limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    float g_eq[7];
    float T_local = temperature[idx];
    float3 vel_local = make_float3(vel_x[idx], vel_y[idx], vel_z[idx]);

    float u_sq = vel_local.x * vel_local.x + vel_local.y * vel_local.y + vel_local.z * vel_local.z;
    float u_mag = sqrtf(u_sq);
    velocity_limit = velocity_limit * 1.0f;

    // Calculate equilibrium distribution for temperature field
    for (int i = 0; i < 7; i++) {
        float ci_dot_u = c_t_dx[i] * vel_local.x + c_t_dy[i] * vel_local.y + c_t_dz[i] * vel_local.z;
        g_eq[i] = w_t[i] * T_local * (1.0f + 3.0f * ci_dot_u);
    }

    float tau = tau_t[idx];
    for (int i = 0; i < 7; i++) {
        g[7*idx + i] = g[7*idx + i] - (1.0f/tau) * (g[7*idx + i] - g_eq[i]);
    }
}

// Temperature field streaming step with adiabatic boundary conditions
__global__ void temperatureStreamingKernel(float* g, float* g_new, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    int idx = z * nx * ny + y * nx + x;

    for (int i = 0; i < 7; i++) {
        int x_next = x + c_t_dx[i];
        int y_next = y + c_t_dy[i];
        int z_next = z + c_t_dz[i];

        // Adiabatic boundary conditions (zero temperature gradient at boundaries)
        if (x_next < 0) {
            // Left boundary: reflect the distribution function
            int reflected_i = -1;
            for (int j = 0; j < 7; j++) {
                if (c_t_dx[j] == -c_t_dx[i] && c_t_dy[j] == c_t_dy[i] && c_t_dz[j] == c_t_dz[i]) {
                    reflected_i = j;
                    break;
                }
            }
            if (reflected_i >= 0) {
                g_new[7*idx + i] = g[7*idx + reflected_i];
            } else {
                g_new[7*idx + i] = g[7*idx + i];
            }
        } else if (x_next >= nx) {
            // Right boundary: reflect the distribution function
            int reflected_i = -1;
            for (int j = 0; j < 7; j++) {
                if (c_t_dx[j] == -c_t_dx[i] && c_t_dy[j] == c_t_dy[i] && c_t_dz[j] == c_t_dz[i]) {
                    reflected_i = j;
                    break;
                }
            }
            if (reflected_i >= 0) {
                g_new[7*idx + i] = g[7*idx + reflected_i];
            } else {
                g_new[7*idx + i] = g[7*idx + i];
            }
        } else if (y_next < 0) {
            // Bottom boundary: reflect the distribution function
            int reflected_i = -1;
            for (int j = 0; j < 7; j++) {
                if (c_t_dx[j] == c_t_dx[i] && c_t_dy[j] == -c_t_dy[i] && c_t_dz[j] == c_t_dz[i]) {
                    reflected_i = j;
                    break;
                }
            }
            if (reflected_i >= 0) {
                g_new[7*idx + i] = g[7*idx + reflected_i];
            } else {
                g_new[7*idx + i] = g[7*idx + i];
            }
        } else if (y_next >= ny) {
            // Top boundary: reflect the distribution function
            int reflected_i = -1;
            for (int j = 0; j < 7; j++) {
                if (c_t_dx[j] == c_t_dx[i] && c_t_dy[j] == -c_t_dy[i] && c_t_dz[j] == c_t_dz[i]) {
                    reflected_i = j;
                    break;
                }
            }
            if (reflected_i >= 0) {
                g_new[7*idx + i] = g[7*idx + reflected_i];
            } else {
                g_new[7*idx + i] = g[7*idx + i];
            }
        } else if (z_next < 0) {
            // Front boundary: reflect the distribution function
            int reflected_i = -1;
            for (int j = 0; j < 7; j++) {
                if (c_t_dx[j] == c_t_dx[i] && c_t_dy[j] == c_t_dy[i] && c_t_dz[j] == -c_t_dz[i]) {
                    reflected_i = j;
                    break;
                }
            }
            if (reflected_i >= 0) {
                g_new[7*idx + i] = g[7*idx + reflected_i];
            } else {
                g_new[7*idx + i] = g[7*idx + i];
            }
        } else if (z_next >= nz) {
            // Back boundary: reflect the distribution function
            int reflected_i = -1;
            for (int j = 0; j < 7; j++) {
                if (c_t_dx[j] == c_t_dx[i] && c_t_dy[j] == c_t_dy[i] && c_t_dz[j] == -c_t_dz[i]) {
                    reflected_i = j;
                    break;
                }
            }
            if (reflected_i >= 0) {
                g_new[7*idx + i] = g[7*idx + reflected_i];
            } else {
                g_new[7*idx + i] = g[7*idx + i];
            }
        } else {
            // Interior points: normal streaming
            int idx_next = z_next * nx * ny + y_next * nx + x_next;
            g_new[7*idx + i] = g[7*idx_next + i];
        }
    }
}

// Temperature field update kernel
__global__ void updateTemperatureKernel(float* g, float* temperature, int nx, int ny, int nz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    float T = 0.0f;
    for (int i = 0; i < 7; i++) {
        T += g[7*idx + i];
    }
    temperature[idx] = T;
}

// CUDA kernel for fluid streaming step
__global__ void streamingKernel(float* f, float* f_new, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    int idx = z * nx * ny + y * nx + x;
    
    // Streaming in each direction
    for (int i = 0; i < 19; i++) {
        int x_new = x + (int)c_dx[i];
        int y_new = y + (int)c_dy[i];
        int z_new = z + (int)c_dz[i];
        
        // Apply open boundary conditions
        if (x_new < 0) x_new = 0;
        if (x_new >= nx) x_new = nx - 1;
        if (y_new < 0) y_new = 0;
        if (y_new >= ny) y_new = ny - 1;
        if (z_new < 0) z_new = 0;
        if (z_new >= nz) z_new = nz - 1;
        
        int idx_new = z_new * nx * ny + y_new * nx + x_new;
        f_new[19 * idx_new + i] = f[19 * idx + i];
    }
}

// CUDA kernel for macroscopic quantity calculation
__global__ void calculateMacroKernel(float* f, float* rho, float* vel_x, float* vel_y, float* vel_z, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    int idx = z * nx * ny + y * nx + x;
    
    float rho_local = 0.0f;
    float3 vel_local = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int i = 0; i < 19; i++) {
        float fi = f[19 * idx + i];
        rho_local += fi;
        vel_local.x += c_dx[i] * fi;
        vel_local.y += c_dy[i] * fi;
        vel_local.z += c_dz[i] * fi;
    }
    
    // Normalize velocity
    if (rho_local > 1e-6f) {
        vel_local.x /= rho_local;
        vel_local.y /= rho_local;
        vel_local.z /= rho_local;
    }

    float u_sq = vel_local.x * vel_local.x + vel_local.y * vel_local.y + vel_local.z * vel_local.z;
    float u_mag = sqrtf(u_sq);

    if (u_mag > 0.0577f) {
        float scale = 0.0577f / u_mag;
        vel_local.x *= scale;
        vel_local.y *= scale;
        vel_local.z *= scale;
    }
    
    // Save results
    rho[idx] = rho_local;
    vel_x[idx] = vel_local.x;
    vel_y[idx] = vel_local.y;
    vel_z[idx] = vel_local.z;
}

// Continuous smoke source injection kernel
__global__ void injectSmokeSourceKernel(float* f, float* g, float* rho, float* temperature, int nx, int ny, int nz, 
                                       float source_radius, float source_density, float source_temperature, 
                                       float injection_rate, int current_step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    int idx = z * nx * ny + y * nx + x;
    
    // Calculate distance from source center
    int center_x = nx / 2;
    int center_y = ny / 8;
    int center_z = nz / 2;
    
    float dx = x - center_x;
    float dy = y - center_y;
    float dz = z - center_z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    
    // Check if point is within source radius
    if (dist < source_radius) {
        // Initialize random number generator
        curandState state;
        curand_init(clock64(), idx + current_step, 0, &state);
        
        // Add smoke density with injection rate
        rho[idx] = injection_rate * (0.8f + 0.2f * curand_uniform(&state));

        // Update fluid distribution function
        for (int i = 0; i < 19; i++) {
            f[19*idx + i] = w[i] * rho[idx];
        }
        
        // Add temperature with injection rate
        temperature[idx] = 300 + (source_temperature) * (1.0f - dist / source_radius) * (0.8f + 0.2f * curand_uniform(&state));
        
        // Update temperature distribution function
        for (int i = 0; i < 7; i++) {
            g[7*idx + i] = w_t[i] * temperature[idx];
        }
    }
}

BoltzmannSolver::BoltzmannSolver(int nx, int ny, int nz, const InitParams& params)
    : nx_(nx), ny_(ny), nz_(nz), d_f_distribution(nullptr), d_g_distribution(nullptr), d_density(nullptr), d_velocity_x(nullptr), d_velocity_y(nullptr), d_velocity_z(nullptr), d_temperature(nullptr), d_tau_f(nullptr), d_tau_t(nullptr), h_density(nullptr), h_temperature(nullptr), init_params_(params) {
    allocateMemory();
    initializeFields();
}

void BoltzmannSolver::allocateMemory() {
    size_t grid_size = nx_ * ny_ * nz_;
    cudaMalloc(&d_f_distribution, grid_size * 19 * sizeof(float));
    cudaMalloc(&d_g_distribution, grid_size * 7 * sizeof(float));
    cudaMalloc(&d_density, grid_size * sizeof(float));
    cudaMalloc(&d_velocity_x, grid_size * sizeof(float));
    cudaMalloc(&d_velocity_y, grid_size * sizeof(float));
    cudaMalloc(&d_velocity_z, grid_size * sizeof(float));
    cudaMalloc(&d_temperature, grid_size * sizeof(float));
    cudaMalloc(&d_tau_f, grid_size * sizeof(float));
    cudaMalloc(&d_tau_t, grid_size * sizeof(float));
    
    h_density = new float[grid_size];
    h_temperature = new float[grid_size];
    h_vel_.resize(grid_size * 3);
    h_rho_.resize(grid_size);
    h_tau_f_.resize(grid_size);
    h_tau_t_.resize(grid_size);
}

void BoltzmannSolver::freeMemory() {
    cudaFree(d_f_distribution);
    cudaFree(d_g_distribution);
    cudaFree(d_density);
    cudaFree(d_velocity_x);
    cudaFree(d_velocity_y);
    cudaFree(d_velocity_z);
    cudaFree(d_temperature);
    cudaFree(d_tau_f);
    cudaFree(d_tau_t);
    delete[] h_density;
    delete[] h_temperature;
}

void BoltzmannSolver::initializeFields() {
    size_t grid_size = nx_ * ny_ * nz_;
    std::vector<float> initial_density(grid_size, 0.00f);
    std::vector<float> initial_velocity(grid_size * 3, 0.0f);
    std::vector<float> initial_f_distribution(grid_size * 19, 0.0f);
    std::vector<float> initial_g_distribution(grid_size * 7, 0.0f);
    std::vector<float> initial_tau_f(grid_size, init_params_.tau_f);
    std::vector<float> initial_tau_t(grid_size, init_params_.tau_t);
    std::vector<float> initial_temperature(grid_size, init_params_.temperature);

    cudaMemcpy(d_f_distribution, initial_f_distribution.data(), grid_size * 19 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_distribution, initial_g_distribution.data(), grid_size * 7 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_density, initial_density.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_x, initial_velocity.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_y, initial_velocity.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_z, initial_velocity.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temperature, initial_temperature.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau_f, initial_tau_f.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau_t, initial_tau_t.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    
    memcpy(h_density, initial_density.data(), grid_size * sizeof(float));
    memcpy(h_temperature, initial_temperature.data(), grid_size * sizeof(float));
    h_tau_f_ = initial_tau_f;
    h_tau_t_ = initial_tau_t;
}

void BoltzmannSolver::streamFluid() {
    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
              (ny_ + block.y - 1) / block.y,
              (nz_ + block.z - 1) / block.z);
    
    // Allocate temporary buffer
    float* d_f_new;
    cudaMalloc(&d_f_new, nx_ * ny_ * nz_ * 19 * sizeof(float));
        
    // Execute streaming kernel
    streamingKernel<<<grid, block>>>(d_f_distribution, d_f_new, nx_, ny_, nz_);
    cudaDeviceSynchronize();
    
    // Copy results
    cudaMemcpy(d_f_distribution, d_f_new, nx_ * ny_ * nz_ * 19 * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Free temporary buffer
    cudaFree(d_f_new);
}

void BoltzmannSolver::streamTemperature() {
    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
              (ny_ + block.y - 1) / block.y,
              (nz_ + block.z - 1) / block.z);
    
    float* d_g_new;
    cudaMalloc(&d_g_new, nx_ * ny_ * nz_ * 7 * sizeof(float));
    cudaMemset(d_g_new, 0, nx_ * ny_ * nz_ * 7 * sizeof(float));
    
    temperatureStreamingKernel<<<grid, block>>>(d_g_distribution, d_g_new, nx_, ny_, nz_);
    cudaDeviceSynchronize();
    
    cudaMemcpy(d_g_distribution, d_g_new, nx_ * ny_ * nz_ * 7 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_g_new);
}

void BoltzmannSolver::collideFluid() {
    int grid_size = nx_ * ny_ * nz_;
    int block_size = 256;
    int num_blocks = (grid_size + block_size - 1) / block_size;
    fluidCollisionKernel<<<num_blocks, block_size>>>(d_f_distribution, d_density, d_velocity_x, d_velocity_y, d_velocity_z, d_tau_f, d_temperature, nx_, ny_, nz_, current_step_, init_params_.velocity_limit, init_params_.beta, init_params_.buoyancy_rand_ratio, init_params_.tau_rand_factor);
    cudaDeviceSynchronize();
}

void BoltzmannSolver::collideTemperature() {
    int grid_size = nx_ * ny_ * nz_;
    int block_size = 256;
    int num_blocks = (grid_size + block_size - 1) / block_size;
    temperatureCollisionKernel<<<num_blocks, block_size>>>(d_g_distribution, d_temperature, d_velocity_x, d_velocity_y, d_velocity_z, d_tau_t, nx_, ny_, nz_, init_params_.velocity_limit);
    cudaDeviceSynchronize();
}

void BoltzmannSolver::updateTemperature() {
    int grid_size = nx_ * ny_ * nz_;
    int block_size = 256;
    int num_blocks = (grid_size + block_size - 1) / block_size;
    updateTemperatureKernel<<<num_blocks, block_size>>>(d_g_distribution, d_temperature, nx_, ny_, nz_);
    cudaDeviceSynchronize();
}

void BoltzmannSolver::injectSmokeSource() {
    
    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
              (ny_ + block.y - 1) / block.y,
              (nz_ + block.z - 1) / block.z);

    injectSmokeSourceKernel<<<grid, block>>>(d_f_distribution, d_g_distribution, d_density, d_temperature,
                                            nx_, ny_, nz_, init_params_.source_radius, init_params_.source_density,
                                            init_params_.source_temperature, init_params_.source_injection_rate, current_step_);
    cudaDeviceSynchronize();
}

void BoltzmannSolver::updateMacroscopic() {
    dim3 block(8, 8, 8);
    dim3 grid((nx_ + block.x - 1) / block.x,
              (ny_ + block.y - 1) / block.y,
              (nz_ + block.z - 1) / block.z);
    
    // Execute macroscopic quantity calculation kernel
    calculateMacroKernel<<<grid, block>>>(d_f_distribution, d_density, d_velocity_x, d_velocity_y, d_velocity_z, nx_, ny_, nz_);
    cudaDeviceSynchronize();
}

void BoltzmannSolver::copyToHost() {
    // Data transfer from GPU to CPU
    cudaMemcpy(h_density, d_density, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_temperature, d_temperature, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Transfer velocity vectors
    std::vector<float> temp_vel_x(nx_ * ny_ * nz_);
    std::vector<float> temp_vel_y(nx_ * ny_ * nz_);
    std::vector<float> temp_vel_z(nx_ * ny_ * nz_);
    
    cudaMemcpy(temp_vel_x.data(), d_velocity_x, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_vel_y.data(), d_velocity_y, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_vel_z.data(), d_velocity_z, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Store velocity vectors in h_vel_
    h_vel_.resize(nx_ * ny_ * nz_ * 3);
    for (int i = 0; i < nx_ * ny_ * nz_; i++) {
        h_vel_[i] = temp_vel_x[i];
        h_vel_[i + nx_ * ny_ * nz_] = temp_vel_y[i];
        h_vel_[i + 2 * nx_ * ny_ * nz_] = temp_vel_z[i];
    }
    
    // Output debug information
    float max_density = 0.0f;
    float avg_density = 0.0f;
    int active_voxels = 0;
    
    for (int i = 0; i < nx_ * ny_ * nz_; i++) {
        if (h_density[i] > 0.1f) {
            active_voxels++;
            max_density = std::max(max_density, h_density[i]);
            avg_density += h_density[i];
        }
    }
    
    if (active_voxels > 0) {
        avg_density /= active_voxels;
    }
}

void BoltzmannSolver::simulate(float dt, int steps) {
    float period = 30.0f;
    float omega = 2.0f * 3.14159265358979323846f / period;
    for (int step = 0; step < steps; ++step) {
        cudaMemcpy(d_density, h_density, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temperature, h_temperature, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyHostToDevice);

        // Fluid simulation steps
        collideFluid();
        collideTemperature();

        streamFluid();
        streamTemperature();

        updateMacroscopic();
        updateTemperature();


        if(init_params_.continuous_source && current_step_ % init_params_.source_injection_interval == 0) {
            injectSmokeSource();
        }

        copyToHost();

        // Display progress
        std::cout << "\rStep " << current_step_ 
                  << " | Active voxels: " << countActiveVoxels() 
                  << " | Max density: " << getMaxDensity() 
                  << " | Avg density: " << getAverageDensity() 
                  << " | Max temperature: " << getMaxTemperature()
                  << " | Avg temperature: " << getAverageTemperature()
                  << " | Avg velocity: " << getAverageVelocity()
                  << " | Max velocity: " << getMaxVelocity()
                  << std::flush;

        current_step_++;
    }
}

BoltzmannSolver::~BoltzmannSolver() {
    freeMemory();
}

// Implementation of statistical functions
int BoltzmannSolver::countActiveVoxels() const {
    int count = 0;
    int grid_size = nx_ * ny_ * nz_;
    for (int i = 0; i < grid_size; i++) {
        if (h_density[i] > 0.001f) {
            count++;
        }
    }
    return count;
}

float BoltzmannSolver::getMaxDensity() const {
    float max_density = 0.0f;
    int grid_size = nx_ * ny_ * nz_;
    for (int i = 0; i < grid_size; i++) {
        max_density = std::max(max_density, h_density[i]);
    }
    return max_density;
}

float BoltzmannSolver::getAverageDensity() const {
    float total_density = 0.0f;
    int active_count = 0;
    int grid_size = nx_ * ny_ * nz_;
    for (int i = 0; i < grid_size; i++) {
        if (h_density[i] > 0.001f) {
            total_density += h_density[i];
            active_count++;
        }
    }
    return active_count > 0 ? total_density / active_count : 0.0f;
}

float BoltzmannSolver::getMaxTemperature() const {
    float max_temp = 0.0f;
    int grid_size = nx_ * ny_ * nz_;
    for (int i = 0; i < grid_size; i++) {
        max_temp = std::max(max_temp, h_temperature[i]);
    }
    return max_temp;
}

float BoltzmannSolver::getAverageTemperature() const {
    float total_temp = 0.0f;
    int active_count = 0;
    int grid_size = nx_ * ny_ * nz_;
    for (int i = 0; i < grid_size; i++) {
        if (h_density[i] > 0.001f) {
            total_temp += h_temperature[i];
            active_count++;
        }
    }
    return active_count > 0 ? total_temp / active_count : 0.0f;
}

float BoltzmannSolver::getAverageVelocity() const {
    float total_velocity = 0.0f;
    int active_count = 0;
    int grid_size = nx_ * ny_ * nz_;
    for (int i = 0; i < grid_size; i++) {
        if (h_density[i] > 0.001f) {
            float vx = h_vel_[3*i];
            float vy = h_vel_[3*i + 1];
            float vz = h_vel_[3*i + 2];
            float velocity_magnitude = sqrtf(vx*vx + vy*vy + vz*vz);
            total_velocity += velocity_magnitude;
            active_count++;
        }
    }
    return active_count > 0 ? total_velocity / active_count : 0.0f;
}

float BoltzmannSolver::getMaxVelocity() const {
    float max_velocity = 0.0f;
    int grid_size = nx_ * ny_ * nz_;
    for (int i = 0; i < grid_size; i++) {
        float vx = h_vel_[3*i];
        float vy = h_vel_[3*i + 1];
        float vz = h_vel_[3*i + 2];
        float velocity_magnitude = sqrtf(vx*vx + vy*vy + vz*vz);
        max_velocity = std::max(max_velocity, velocity_magnitude);
    }
    return max_velocity;
}
