#include "scene.h"

std::chrono::time_point<std::chrono::system_clock> start;

void start_timer()
{
    start = std::chrono::system_clock::now();
}

void end_timer()
{
    std::chrono::time_point<std::chrono::system_clock> end;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> delta_time = end - start;
    std::chrono::duration<double> delta_time_sec = end - start;

    std::cout << "\e[7mFrame duration:\t" << std::setw(5) << delta_time.count() << " ms"<< "\tFramerate:\t" << std::setw(5) << 1 / delta_time_sec.count() << " fps\r";
    std::flush(std::cout);
}

Scene::Scene(Camera* cam)
: p_cam(cam),
  d_buffer(nullptr),
  d_random(nullptr),
  d_raydirs(nullptr),
  d_shapes(nullptr)
{
    // Initialize shapes here.
    std::vector<shape_t> shapes;
    shapes.push_back({ { Vector(0,0,-3).gpu_type(), .2 }, SPHERE, 0.f, { 255, 0,   0 } });
    shapes.push_back({ { Vector(2,0,-3).gpu_type(), .2 }, SPHERE, 1.f, { 255, 255, 0 } });

    SAFE( cudaMalloc(&d_buffer,  p_cam->size() * sizeof(float4)) );
    SAFE( cudaMalloc(&d_random,  p_cam->size() * sizeof(float4)) );
    SAFE( cudaMalloc(&d_raydirs, p_cam->size() * sizeof(ray_t)) );
    SAFE( cudaMalloc(&d_shapes,  shapes.size() * sizeof(shape_t)) );
    SAFE( cudaMemcpy( d_shapes,  shapes.data(), shapes.size() * sizeof(shape_t), cudaMemcpyHostToDevice) );
}

Scene::~Scene()
{
    SAFE( cudaFree(d_shapes) );
    SAFE( cudaFree(d_raydirs) );
    SAFE( cudaFree(d_random) );
    SAFE( cudaFree(d_buffer) );
}

void Scene::render(color_t* buffer)
{
    start_timer();
    cudaproduceray(p_cam->gpu_type(), d_raydirs);
    cudapathtrace (p_cam->gpu_type(), (color_t*)d_buffer, d_random, d_raydirs, d_shapes, 2); // FIXME: hardcoded shape size

    SAFE( cudaMemcpy( buffer, d_buffer, p_cam->size() * sizeof(color_t), cudaMemcpyDeviceToHost) );
    end_timer();
}
