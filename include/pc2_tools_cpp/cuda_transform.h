#ifndef PC2_TOOLS_CPP_CUDA_TRANSFORM_H
#define PC2_TOOLS_CPP_CUDA_TRANSFORM_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
// float3 is already defined in CUDA headers
#else
// CPU fallback: define float3 if CUDA not available
struct float3 {
    float x, y, z;
};
#endif

// CUDA wrapper function
extern "C" void transformPointsCUDA(
    const float3* h_points_in,
    float3* h_points_out,
    const float* R,  // 3x3 rotation matrix (row-major)
    const float3 T,  // Translation vector
    int n_points,
    int skip_points,
    int* n_valid_out
);

#endif  // PC2_TOOLS_CPP_CUDA_TRANSFORM_H

