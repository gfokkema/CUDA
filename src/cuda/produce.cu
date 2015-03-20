#include "gpu_kernels.h"

__global__
void
produceray(__const__ camera_t cam,
           ray_t* d_raydirs,
           color_t* d_buffer)
{
    unsigned xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yi = blockIdx.y * blockDim.y + threadIdx.y;

    float x = 2 * (xi + .5) / cam.width  - 1;
    float y = 2 * (yi + .5) / cam.height - 1;

    int idx = yi * cam.width + xi;
    d_raydirs[idx].pos = cam.pos + cam.dir + x * cam.right + y * cam.up;
    d_raydirs[idx].dir = normalize(d_raydirs[idx].pos - cam.pos);
    d_buffer[idx] = (color_t){ 0, 0, 0 };
}

__host__
int
cudaproduceray(camera_t cam,
               ray_t*   d_raydirs,
               color_t* d_buffer)
{
    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    produceray <<< numblocks,threadsperblock >>> (cam, d_raydirs, d_buffer);
    CHECK_ERROR("Launching produce kernel");

    return 0;
}
