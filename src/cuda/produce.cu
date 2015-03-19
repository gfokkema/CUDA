#include "device_util.cuh"
#include "host_kernels.cuh"

__global__
void produceray(__const__ camera_t cam, ray_t* raydirs) {
    unsigned xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yi = blockIdx.y * blockDim.y + threadIdx.y;

    float x = 2 * (xi + .5) / cam.width  - 1;
    float y = 2 * (yi + .5) / cam.height - 1;

    int idx = yi * cam.width + xi;
    raydirs[idx].pos = cam.pos + cam.dir + x * cam.right + y * cam.up;
    raydirs[idx].dir = normalize(raydirs[idx].pos - cam.pos);
}

__host__
int cudaproduceray(camera_t cam,
                   ray_t* d_raydirs)
{
    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    produceray <<< numblocks,threadsperblock >>> (cam, d_raydirs);
    CHECK_ERROR("Launching produce kernel");

    return 0;
}
