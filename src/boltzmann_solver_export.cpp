#include "../include/boltzmann_solver.hpp"
#include "../include/vdb_exporter.hpp"
#include <cmath>
#include <iostream>

void BoltzmannSolver::initialize() {
    // 初期状態の設定（煙の発生源）
    // for (int z = 0; z < nz_; z++) {
    //     for (int y = 0; y < ny_; y++) {
    //         for (int x = 0; x < nx_; x++) {
    //             // 中心付近に煙を配置
    //             float dx = x - nx_/2;
    //             float dy = y - ny_/2;
    //             float dz = z - nz_/2;
    //             float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    //             if (dist < 10.0f) {
    //                 h_density[z * nx_ * ny_ + y * nx_ + x] = 2.0f;
    //             }
    //         }
    //     }
    // }
    // GPUメモリにコピー
    cudaMemcpy(d_density, h_density, nx_ * ny_ * nz_ * sizeof(float), cudaMemcpyHostToDevice);
}

void BoltzmannSolver::exportToVDB(const char* filename, VDBExporter* exporter) {
    // GPUからCPUへデータをコピー
    copyToHost();
    // OpenVDBファイルに出力
    if (exporter) {
        exporter->exportToVDB(filename, h_rho_.data(), h_vel_.data());
    }
}

extern "C" {
    // Set initial state (smoke source)
    void initializeSmokeSource(float* density, float* temperature, int nx, int ny, int nz) {
        int center_x = nx / 2;
        int center_y = ny / 8;
        int center_z = nz / 2;
        float source_radius = 6.0f;
        float source_density = 2.1f;

        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    float dx = x - center_x;
                    float dy = y - center_y;
                    float dz = z - center_z;
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    int idx = z * nx * ny + y * nx + x;

                    if (dist < source_radius) {
                        // Place smoke near center
                        density[idx] = source_density;
                        temperature[idx] = 500.0f;
                    }
                }
            }
        }
    }

    void exportSimulationFrame(const char* filename, const float* density, const float* velocity, int nx, int ny, int nz) {
        // Copy to GPU memory
        VDBExporter exporter(nx, ny, nz);
        
        // Copy data from GPU to CPU
        exporter.exportToVDB(filename, density, velocity);
        
        // Output to OpenVDB file
        std::cout << "Exported frame to: " << filename << std::endl;
    }
} 