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
  d_materials(nullptr),
  d_raydirs(nullptr),
  d_shapes(nullptr)
{
    // mat_t { { r, g, b }, emit, n, brdf, type }
    std::vector<mat_t> materials;
    materials.push_back({ { 1, 1, 0 }, 1, 1, 0, DIFFUSE });             // 0: WHITE LIGHTING
    materials.push_back({ { 1, 1, 1 }, 0, 1, 0, MIRROR });              // 1: REFLECTIVE
    materials.push_back({ { 1, 1, 1 }, 0, 1, 0, DIFFUSE });             // 2: DIFFUSE WHITE
    materials.push_back({ { 0, 0, 1 }, 0, 1, 0, DIFFUSE });             // 3: DIFFUSE BLUE
    materials.push_back({ { 1, 0, 0 }, 0, 1, 0, DIFFUSE });             // 4: DIFFUSE RED

    // Initialize shapes here.
    // shape_t { { origin, radius }, matidx, type }
    std::vector<shape_t> shapes;
    shapes.push_back({ { Vector(0,0,-3).gpu_type(), .2 }, 0, SPHERE });
    shapes.push_back({ { Vector(2,0,-3).gpu_type(), 1 }, 1, SPHERE });
    shapes.push_back({ { Vector(0,0,-50).gpu_type(), 45 }, 2, SPHERE }); // BACK: WHITE
    shapes.push_back({ { Vector(-50,0,0).gpu_type(), 45 }, 4, SPHERE }); // LEFT:  RED
    shapes.push_back({ { Vector( 50,0,0).gpu_type(), 45 }, 3, SPHERE }); // RIGHT: BLUE
    shapes.push_back({ { Vector( 0,50,0).gpu_type(), 45 }, 2, SPHERE }); // CEILING: WHITE

    SAFE( cudaMalloc(&d_buffer,    p_cam->size()    * sizeof(float4)) );
    SAFE( cudaMalloc(&d_random,    p_cam->size()    * sizeof(float4)) );
    SAFE( cudaMalloc(&d_raydirs,   p_cam->size()    * sizeof(ray_t)) );
    SAFE( cudaMalloc(&d_result,    p_cam->size()    * sizeof(float4)) );
    SAFE( cudaMalloc(&d_materials, materials.size() * sizeof(mat_t)) )
    SAFE( cudaMalloc(&d_shapes,    shapes.size()    * sizeof(shape_t)) );
    SAFE( cudaMemcpy( d_materials, materials.data(), materials.size() * sizeof(mat_t),   cudaMemcpyHostToDevice) );
    SAFE( cudaMemcpy( d_shapes,    shapes.data(),    shapes.size()    * sizeof(shape_t), cudaMemcpyHostToDevice) );
}

Scene::~Scene()
{
    SAFE( cudaFree(d_shapes) );
    SAFE( cudaFree(d_raydirs) );
    SAFE( cudaFree(d_materials) );
    SAFE( cudaFree(d_random) );
    SAFE( cudaFree(d_buffer) );
}

void Scene::render(color_t* buffer)
{
    start_timer();
//    for (int i = 0; i < 100; i++) {
    cudaproduceray(p_cam->gpu_type(), d_raydirs, d_result);
    cudapathtrace (p_cam->gpu_type(), d_result, d_random, d_materials, d_raydirs, d_shapes, 6); // FIXME: hardcoded shape size
//    }
    cudargbtoint  (p_cam->gpu_type(), d_result, d_buffer);

    SAFE( cudaMemcpy( buffer, d_buffer, p_cam->size() * sizeof(color_t), cudaMemcpyDeviceToHost) );
    end_timer();
}
