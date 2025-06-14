#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <vector>

class VDBExporter {
public:
    VDBExporter(int nx, int ny, int nz);
    ~VDBExporter();

    // Output simulation results to OpenVDB file
    void exportToVDB(const char* filename, const float* density, const float* velocity);

private:
    int nx_, ny_, nz_;
    openvdb::FloatGrid::Ptr density_grid_;
    openvdb::Vec3fGrid::Ptr velocity_grid_;

    void initializeGrids();
    void updateGrids(const float* density, const float* velocity);
}; 