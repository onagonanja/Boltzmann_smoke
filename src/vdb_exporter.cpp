#include "vdb_exporter.hpp"
#include <iostream>

VDBExporter::VDBExporter(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz) {
    initializeGrids();
}

VDBExporter::~VDBExporter() {
    // OpenVDB's grids are managed by smart pointers, so explicit freeing is not required
}

void VDBExporter::initializeGrids() {
    // Initialize density grid
    density_grid_ = openvdb::FloatGrid::create(0.0f);
    density_grid_->setName("density");  // Change to name easily recognized by Blender
    density_grid_->setTransform(openvdb::math::Transform::createLinearTransform(0.1));  // Set scale to 0.1

    // Initialize velocity grid
    velocity_grid_ = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
    velocity_grid_->setName("velocity");  // Change to name easily recognized by Blender
    velocity_grid_->setTransform(openvdb::math::Transform::createLinearTransform(0.1));  // Set scale to 0.1
}

void VDBExporter::updateGrids(const float* density, const float* velocity) {
    // Create accessors
    openvdb::FloatGrid::Accessor density_accessor = density_grid_->getAccessor();
    openvdb::Vec3fGrid::Accessor velocity_accessor = velocity_grid_->getAccessor();

    // Debug counter
    int active_voxels = 0;
    float max_density = 0.0f;
    
    // Update grids
    for (int z = 0; z < nz_; ++z) {
        for (int y = 0; y < ny_; ++y) {
            for (int x = 0; x < nx_; ++x) {
                int idx = z * nx_ * ny_ + y * nx_ + x;
                openvdb::Coord coord(x, y, z);
                
                // Update density (increase scale for better visibility)
                float scaled_density = density[idx] * 10.0f;  // Multiply density by 10
                if (scaled_density > 0.001f) {  // Lower threshold to 0.001
                    density_accessor.setValue(coord, scaled_density);
                    active_voxels++;
                    max_density = std::max(max_density, scaled_density);
                }
                
                // Update velocity (adjust scale)
                openvdb::Vec3f vel(
                    velocity[3*idx] * 0.1f,  // Multiply velocity by 0.1
                    velocity[3*idx + 1] * 0.1f,
                    velocity[3*idx + 2] * 0.1f
                );
                if (scaled_density > 0.001f) {  // Lower threshold to 0.001
                    velocity_accessor.setValue(coord, vel);
                }
            }
        }
    }
    
    // Output debug information
    // std::cout << "VDB output debug information:" << std::endl;
    // std::cout << "Active voxels: " << active_voxels << std::endl;
    // std::cout << "Max density: " << max_density << std::endl;
    // std::cout << "Grid size: " << nx_ << "x" << ny_ << "x" << nz_ << std::endl;
}

void VDBExporter::exportToVDB(const char* filename, const float* density, const float* velocity) {
    // Update grids
    updateGrids(density, velocity);
    
    // Write to file
    openvdb::io::File file(filename);
    openvdb::GridPtrVec grids;
    grids.push_back(density_grid_);
    grids.push_back(velocity_grid_);
    file.write(grids);
    file.close();
} 