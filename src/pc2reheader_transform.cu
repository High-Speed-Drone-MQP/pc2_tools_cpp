#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

// CUDA kernel to transform points: R * p + T
__global__ void transformPointsKernel(
    const float3* points_in,
    float3* points_out,
    const float* R,  // 3x3 rotation matrix (row-major)
    const float3 T,  // Translation vector
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        float3 p = points_in[idx];
        
        // Check for NaN/Inf
        if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
            // Mark as invalid (will be filtered out)
            points_out[idx].x = __int_as_float(0x7fffffff);  // NaN marker
            points_out[idx].y = __int_as_float(0x7fffffff);
            points_out[idx].z = __int_as_float(0x7fffffff);
            return;
        }
        
        // Transform: R * p + T
        points_out[idx].x = R[0] * p.x + R[1] * p.y + R[2] * p.z + T.x;
        points_out[idx].y = R[3] * p.x + R[4] * p.y + R[5] * p.z + T.y;
        points_out[idx].z = R[6] * p.x + R[7] * p.y + R[8] * p.z + T.z;
    }
}

// CUDA kernel to downsample and filter points
__global__ void downsampleAndFilterKernel(
    const float3* points_in,
    float3* points_out,
    int* valid_count,
    int skip_points,
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use atomic counter for valid points
    __shared__ int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();
    
    if (idx < n_points && (idx % skip_points == 0)) {
        float3 p = points_in[idx];
        
        // Check for valid (finite) points
        if (isfinite(p.x) && isfinite(p.y) && isfinite(p.z)) {
            int out_idx = atomicAdd(&shared_count, 1);
            if (out_idx < n_points / skip_points + 100) {  // Safety bound
                points_out[out_idx] = p;
            }
        }
    }
    
    __syncthreads();
    
    // Update global counter
    if (threadIdx.x == 0) {
        atomicAdd(valid_count, shared_count);
    }
}

// Combined kernel: downsample, filter, and transform in one pass
// Uses compact array - first pass marks valid, second pass compacts
__global__ void downsampleFilterTransformKernel(
    const float3* points_in,
    float3* points_out,
    int* valid_flags,  // Output: 1 if valid, 0 if invalid
    const float* R,
    const float3 T,
    int skip_points,
    int n_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_points && (idx % skip_points == 0)) {
        float3 p = points_in[idx];
        
        // Check for valid (finite) points
        if (isfinite(p.x) && isfinite(p.y) && isfinite(p.z)) {
            // Transform point
            float3 p_transformed;
            p_transformed.x = R[0] * p.x + R[1] * p.y + R[2] * p.z + T.x;
            p_transformed.y = R[3] * p.x + R[4] * p.y + R[5] * p.z + T.y;
            p_transformed.z = R[6] * p.x + R[7] * p.y + R[8] * p.z + T.z;
            
            // Check transformed point is still valid
            if (isfinite(p_transformed.x) && isfinite(p_transformed.y) && isfinite(p_transformed.z)) {
                points_out[idx / skip_points] = p_transformed;
                valid_flags[idx / skip_points] = 1;
            } else {
                valid_flags[idx / skip_points] = 0;
            }
        } else {
            valid_flags[idx / skip_points] = 0;
        }
    } else if (idx < n_points) {
        // Not a sampled point
        if (idx / skip_points < (n_points / skip_points + 100)) {
            valid_flags[idx / skip_points] = 0;
        }
    }
}

// Wrapper function to call CUDA kernel
extern "C" void transformPointsCUDA(
    const float3* h_points_in,
    float3* h_points_out,
    const float* R,
    const float3 T,
    int n_points,
    int skip_points,
    int* n_valid_out
) {
    // Allocate device memory
    float3* d_points_in;
    float3* d_points_out;
    float* d_R;
    float3 d_T = T;
    int* d_valid_flags;
    
    size_t points_size = n_points * sizeof(float3);
    int output_size = (n_points / skip_points + 100);
    size_t output_bytes = output_size * sizeof(float3);
    size_t flags_bytes = output_size * sizeof(int);
    
    cudaMalloc(&d_points_in, points_size);
    cudaMalloc(&d_points_out, output_bytes);
    cudaMalloc(&d_R, 9 * sizeof(float));
    cudaMalloc(&d_valid_flags, flags_bytes);
    
    // Copy data to device
    cudaMemcpy(d_points_in, h_points_in, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_valid_flags, 0, flags_bytes);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n_points + threads_per_block - 1) / threads_per_block;
    
    downsampleFilterTransformKernel<<<blocks, threads_per_block>>>(
        d_points_in, d_points_out, d_valid_flags, d_R, d_T, skip_points, n_points
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling - fall back to CPU
        *n_valid_out = 0;
        cudaFree(d_points_in);
        cudaFree(d_points_out);
        cudaFree(d_R);
        cudaFree(d_valid_flags);
        return;
    }
    
    // Copy results back
    int* h_valid_flags = (int*)malloc(flags_bytes);
    float3* h_points_temp = (float3*)malloc(output_bytes);
    
    if (!h_valid_flags || !h_points_temp) {
        // Malloc failed - cleanup and return
        *n_valid_out = 0;
        if (h_valid_flags) free(h_valid_flags);
        if (h_points_temp) free(h_points_temp);
        cudaFree(d_points_in);
        cudaFree(d_points_out);
        cudaFree(d_R);
        cudaFree(d_valid_flags);
        return;
    }
    
    // Copy from device to host (check for errors)
    err = cudaMemcpy(h_valid_flags, d_valid_flags, flags_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_valid_flags);
        free(h_points_temp);
        cudaFree(d_points_in);
        cudaFree(d_points_out);
        cudaFree(d_R);
        cudaFree(d_valid_flags);
        *n_valid_out = 0;
        return;
    }
    
    err = cudaMemcpy(h_points_temp, d_points_out, output_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_valid_flags);
        free(h_points_temp);
        cudaFree(d_points_in);
        cudaFree(d_points_out);
        cudaFree(d_R);
        cudaFree(d_valid_flags);
        *n_valid_out = 0;
        return;
    }
    
    // Compact valid points on CPU (simple and fast)
    int valid_count = 0;
    for (int i = 0; i < output_size; ++i) {
        if (h_valid_flags[i] == 1) {
            h_points_out[valid_count++] = h_points_temp[i];
        }
    }
    *n_valid_out = valid_count;
    
    // Free temporary host memory
    free(h_valid_flags);
    free(h_points_temp);
    
    // Free device memory
    cudaFree(d_points_in);
    cudaFree(d_points_out);
    cudaFree(d_R);
    cudaFree(d_valid_flags);
}

