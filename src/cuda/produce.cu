#include <cstdio>
#include <cuda/gpu_kernels.h>
#include <cuda/gpu_types.h>
#include <cuda/gpu_vector.h>

__device__
float
clamp(float a)
{
    return a > 1 ? 1 : a < 0 ? 0 : a;
}

__global__
void
produceray(__const__ state_t  state,
           __const__ scene_t  scene)
{
    __const__ camera_t& cam = scene.camera;
    unsigned xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = yi * scene.camera.width + xi;

    float x = (2 * (xi + .5) / cam.width  - 1) * cam.ratio;
    float y = (2 * (yi + .5) / cam.height - 1);

    // Initialize rays
    state.rays[idx].pos = cam.pos;
    state.rays[idx].dir = normalize(cam.dir + x * cam.right + y * cam.up);

    // Initialize result buffers
    state.factor[idx]  = (float4){ 1, 1, 1, 0 };
    state.result[idx]  = (float4){ 0, 0, 0, 0 };
}

__global__
void
rgbtoint(__const__ state_t  state,
         __const__ scene_t  scene,
         __const__ output_t output,
         short     samplecount)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx = (yi * scene.camera.width + xi);

    // Accumulate samples on the film, average them with samplecount
    output.film[idx] = output.film[idx] + state.result[idx];

    output.output[idx].r = (int)(clamp(output.film[idx].x / samplecount) * 255);
    output.output[idx].g = (int)(clamp(output.film[idx].y / samplecount) * 255);
    output.output[idx].b = (int)(clamp(output.film[idx].z / samplecount) * 255);
}

__host__
void
cudaproduceray(dims_t   dims,
               state_t  state,
               scene_t  scene)
{
    produceray <<< dims.blocks, dims.threads >>> (state, scene);
}

__host__
void
cudargbtoint(dims_t   dims,
             state_t  state,
             scene_t  scene,
             output_t output,
             short    samplecount)
{
    rgbtoint <<< dims.blocks, dims.threads >>> (state, scene, output, samplecount);
}
