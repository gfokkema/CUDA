#include "gpu_kernels.h"

__global__
void
produceray(__const__ camera_t cam,
           ray_t*  d_raydirs,
           float4*  d_factor,
           float4* d_result,
           float4* d_film,
           short samplecount)
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
    d_factor[idx]  = (float4){ 1, 1, 1, 0};
    d_result[idx]  = (float4){ 0, 0, 0, 0};
    if (samplecount == 0)
        d_film[idx] = (float4){ 0, 0, 0, 0};
}

__host__
int
cudaproduceray(camera_t cam,
               ray_t*   d_raydirs,
               float4*  d_factor,
               float4*  d_result,
               float4*  d_film,
               short    samplecount)
{
    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    produceray <<< numblocks,threadsperblock >>> (
        cam, d_raydirs, d_factor, d_result, d_film, samplecount
    );
    CHECK_ERROR("Launching produce kernel");

    return 0;
}

__device__
float
clamp(float a)
{
    return a > 1 ? 1 : a < 0 ? 0 : a;
}

__global__
void
rgbtoint(camera_t cam,
         float4*  d_result,
         float4*  d_film,
         short    samplecount,
         color_t* d_buffer)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx = (yi * cam.width + xi);

    // Accumulate samples on the film, average them with samplecount
    d_film[idx] = d_film[idx] + d_result[idx];

    d_buffer[idx].r = (int)(clamp(d_film[idx].x / samplecount) * 255);
    d_buffer[idx].g = (int)(clamp(d_film[idx].y / samplecount) * 255);
    d_buffer[idx].b = (int)(clamp(d_film[idx].z / samplecount) * 255);
}

int cudargbtoint(camera_t        cam,
                 float4*         d_result,
                 float4*         d_film, short samplecount,
                 color_t*        d_buffer)
{
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    rgbtoint <<< numblocks,threadsperblock >>> (
        cam, d_result, d_film, samplecount, d_buffer
    );
    CHECK_ERROR("Launching rgbtoint kernel");

    return 0;
}
