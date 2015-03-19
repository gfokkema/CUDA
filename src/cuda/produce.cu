#include "device_util.cuh"
#include "host_kernels.cuh"

__global__
void produceray(__const__ camera_t cam, float4* raydirs) {
    unsigned xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yi = blockIdx.y * blockDim.y + threadIdx.y;

    float x = (xi + .5) / cam.width - 0.5;
    float y = (yi + .5) / cam.height - 0.5;

    raydirs[yi * cam.width + xi] = normalize(x * cam.right + y * cam.up + cam.dir);
}

__host__
int cudaproduceray(camera_t cam,
                   float4*& d_raydirs)
{
    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    produceray <<< numblocks,threadsperblock >>> (cam, d_raydirs);
    CHECK_ERROR("Launching produce kernel");

    return 0;
}
