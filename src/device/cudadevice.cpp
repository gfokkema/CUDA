#include <common.h>
#include <device/cudadevice.h>
#include <scene.h>

#include <chrono>
#include <iomanip>
#include <iostream>

CudaDevice::CudaDevice(int pixels, int matsize, int shapesize)
: d_raydirs(nullptr),
  d_random(nullptr),
  d_factor(nullptr),
  d_result(nullptr),
  d_film(nullptr),
  d_output(nullptr),
  d_mats(nullptr),
  d_shapes(nullptr),
  matsize(matsize),
  shapesize(shapesize)
{
    // Initialize CUDA with a NOOP call
    if (cudaFree(0) != cudaSuccess)
    {
        throw std::runtime_error("Failed to initialize CUDA.");
    }

    // RANDOM GENERATOR
    SAFE_RAND(curandCreateGenerator(&d_generator, CURAND_RNG_PSEUDO_DEFAULT));
    SAFE_RAND(curandSetPseudoRandomGeneratorSeed(d_generator, time(NULL)));

    // PER PIXEL BUFFERS
    SAFE(cudaMalloc(&d_random, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_raydirs, pixels * sizeof(ray_t)));
    SAFE(cudaMalloc(&d_factor, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_result, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_film, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_output, pixels * sizeof(color_t)));

    // SCENE DESCRIPTION
    SAFE(cudaMalloc(&d_mats, matsize * sizeof(mat_t)));
    SAFE(cudaMalloc(&d_shapes, shapesize * sizeof(shape_t)));
}

CudaDevice::~CudaDevice()
{
    SAFE_RAND(curandDestroyGenerator(d_generator));
    SAFE(cudaFree(d_raydirs));
    SAFE(cudaFree(d_random));
    SAFE(cudaFree(d_factor));
    SAFE(cudaFree(d_result));
    SAFE(cudaFree(d_film));
    SAFE(cudaFree(d_output));
    SAFE(cudaFree(d_mats));
    SAFE(cudaFree(d_shapes));
}

void
CudaDevice::copy(std::vector<mat_t> materials, std::vector<shape_t> shapes)
{
    SAFE(cudaMemcpy(d_mats, materials.data(), materials.size() * sizeof(mat_t), cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_shapes, shapes.data(), shapes.size() * sizeof(shape_t), cudaMemcpyHostToDevice));
}

double
CudaDevice::producerays(camera_t & camera, unsigned camsize, unsigned sample)
{
    SAFE_RAND(curandGenerateUniform(d_generator, (float *)d_random, camsize * sizeof(float4)));
    cudaproduceray(camera, d_raydirs, d_factor, d_result, d_film, sample);

    return 0.f;
}

double
CudaDevice::pathtrace(camera_t & camera)
{
    scene_t scene = { 8, 0, camera, d_shapes, d_mats }; // FIXME: hardcoded shape size
    cudapathtrace(scene, d_raydirs, d_random, d_factor, d_result);

    return 0.f;
}

double
CudaDevice::rgbtoint(camera_t & camera, unsigned sample)
{
    cudargbtoint(camera, d_result, d_film, d_output, sample);

    return 0.f;
}

double
CudaDevice::write(color_t * buffer, unsigned size)
{
    SAFE(cudaMemcpy(buffer, d_output, size * sizeof(color_t), cudaMemcpyDeviceToHost));

    return 0.f;
}
