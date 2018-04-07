#include <common.h>
#include <device/cudadevice.h>
#include <scene.h>

#include <chrono>
#include <iomanip>
#include <iostream>

static int sample;
std::chrono::time_point<std::chrono::system_clock> start;

void
start_timer()
{
    start = std::chrono::system_clock::now();
}

void
end_timer()
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> delta_time = end - start;
    std::cout << "\e[7mFrame duration:\t" << std::setw(5) << delta_time.count()
              << " ms" << "\tFramerate:\t" << std::setw(5)
              << 1000 / delta_time.count() << " fps\r";
    std::flush(std::cout);
}

CudaDevice::CudaDevice(int pixels, int matsize, int shapesize)
: d_factor(nullptr),
  d_random(nullptr),
  d_raydirs(nullptr),
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
    SAFE(cudaMalloc(&d_output, pixels * sizeof(color_t)));
    SAFE(cudaMalloc(&d_random, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_raydirs, pixels * sizeof(ray_t)));
    SAFE(cudaMalloc(&d_factor, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_result, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_film, pixels * sizeof(float4)));

    // SCENE DESCRIPTION
    SAFE(cudaMalloc(&d_mats, matsize * sizeof(mat_t)));
    SAFE(cudaMalloc(&d_shapes, shapesize * sizeof(shape_t)));
}

CudaDevice::~CudaDevice()
{
    SAFE_RAND(curandDestroyGenerator(d_generator));
    SAFE(cudaFree(d_factor));
    SAFE(cudaFree(d_raydirs));
    SAFE(cudaFree(d_random));
    SAFE(cudaFree(d_film));
    SAFE(cudaFree(d_output));
    SAFE(cudaFree(d_mats));
    SAFE(cudaFree(d_shapes));
}

double
CudaDevice::render(color_t * buffer, Scene * scene)
{
    int camsize = scene->camera()->size();
    camera_t cam = scene->camera()->gpu_type();
    scene_t _scene = { 8, 0, cam, d_shapes, d_mats }; // FIXME: hardcoded shape size

    //    start_timer();
    SAFE(cudaMemcpy(d_mats, scene->materials().data(), matsize * sizeof(mat_t), cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_shapes, scene->shapes().data(), shapesize * sizeof(shape_t), cudaMemcpyHostToDevice));
    SAFE_RAND(curandGenerateUniform(d_generator, (float *)d_random, camsize * sizeof(float4)));

    cudaproduceray(cam, d_raydirs, d_factor, d_result, d_film, sample);
    cudapathtrace(_scene, d_raydirs, d_factor, d_result, d_random);
    cudargbtoint(cam, d_result, d_film, sample, d_output);

    SAFE(cudaMemcpy(buffer, d_output, camsize * sizeof(color_t), cudaMemcpyDeviceToHost));
    printf("%d\n", sample++);
    //    end_timer();
    return 0.f;
}
