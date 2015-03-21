#include "gpu_kernels.h"

__global__
void
produceray(__const__ camera_t cam,
           ray_t*  d_raydirs,
           float4* d_reflect,
           float4* d_result)
{
    unsigned xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = yi * cam.width + xi;

    float x = 2 * (xi + .5) / cam.width  - 1;
    float y = 2 * (yi + .5) / cam.height - 1;

    // Initialize rays
    d_raydirs[idx].pos = cam.pos + cam.dir + x * cam.right + y * cam.up;
    d_raydirs[idx].dir = normalize(d_raydirs[idx].pos - cam.pos);

    // Initialize result buffers
    d_reflect[idx] = (float4){ 1, 1, 1, 0};
    d_result[idx]  = (float4){ 0, 0, 0, 0};
}

__host__
int
cudaproduceray(camera_t cam,
               ray_t*   d_raydirs,
               float4*  d_reflect,
               float4*  d_result)
{
    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    produceray <<< numblocks,threadsperblock >>> (cam,
                                                  d_raydirs,
                                                  d_reflect,
                                                  d_result);
    CHECK_ERROR("Launching produce kernel");

    return 0;
}
