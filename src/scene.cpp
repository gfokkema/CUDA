#include "scene.h"

std::chrono::time_point<std::chrono::system_clock> start;

static int sample;

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
  d_factor(nullptr),
  d_shapes(nullptr)
{
    // mat_t { { r, g, b }, emit, n, type }
    std::vector<mat_t> materials;
    materials.push_back({ { 0, 0, 0 }, { 1, 1, 1 }, DIFFUSE });          // 0: WHITE LIGHTING
    materials.push_back({ { 0, 0, 0 }, { 0, 0, 0 }, MIRROR  });          // 1: REFLECTIVE
    materials.push_back({ {.6,.6,.6 }, { 0, 0, 0 }, DIFFUSE });          // 2: DIFFUSE WHITE
    materials.push_back({ {.2,.2, 1 }, { 0, 0, 0 }, DIFFUSE });          // 3: DIFFUSE BLUE
    materials.push_back({ { 1,.2,.2 }, { 0, 0, 0 }, DIFFUSE });          // 4: DIFFUSE RED
    materials.push_back({ {.2, 1,.2 }, { 0, 0, 0 }, DIFFUSE });          // 5: DIFFUSE GREEN

    // Initialize shapes here.
    // shape_t { { origin, radius }, matidx, type }
    std::vector<shape_t> shapes;
    shapes.push_back({ { Vector(   -.5,     0,  -1.5).gpu_type(),  .3 }, 0, SPHERE });
    shapes.push_back({ { Vector(    .5,     0,  -1.5).gpu_type(),  .3 }, 1, SPHERE });
    shapes.push_back({ { Vector(     0, 1e5+1,     0).gpu_type(), 1e5 }, 2, SPHERE }); // CEILING: WHITE
    shapes.push_back({ { Vector(     0,-1e5-1,     0).gpu_type(), 1e5 }, 2, SPHERE }); // FLOOR: WHITE
    shapes.push_back({ { Vector( 1e5+1,     0,     0).gpu_type(), 1e5 }, 3, SPHERE }); // RIGHT: BLUE
    shapes.push_back({ { Vector(-1e5-1,     0,     0).gpu_type(), 1e5 }, 4, SPHERE }); // LEFT:  RED
    shapes.push_back({ { Vector(     0,     0,-1e5-2).gpu_type(), 1e5 }, 2, SPHERE }); // BACK: GREEN
    shapes.push_back({ { Vector(     0,     0, 1e5+0).gpu_type(), 1e5 }, 2, SPHERE }); // BEHIND: WHITE

    // PER PIXEL BUFFERS
    SAFE( cudaMalloc(&d_buffer,  p_cam->size() * sizeof(color_t)) );
    SAFE( cudaMalloc(&d_random,  p_cam->size() * sizeof(float4))  );
    SAFE( cudaMalloc(&d_raydirs, p_cam->size() * sizeof(ray_t))   );
    SAFE( cudaMalloc(&d_factor,  p_cam->size() * sizeof(float4))  );
    SAFE( cudaMalloc(&d_result,  p_cam->size() * sizeof(float4))  );
    SAFE( cudaMalloc(&d_film,    p_cam->size() * sizeof(float4))  );

    // SCENE DESCRIPTION
    SAFE( cudaMalloc(&d_materials, materials.size() * sizeof(mat_t))   );
    SAFE( cudaMalloc(&d_shapes,    shapes.size()    * sizeof(shape_t)) );
    SAFE( cudaMemcpy( d_materials, materials.data(), materials.size() * sizeof(mat_t),   cudaMemcpyHostToDevice) );
    SAFE( cudaMemcpy( d_shapes,    shapes.data(),    shapes.size()    * sizeof(shape_t), cudaMemcpyHostToDevice) );
}

Scene::~Scene()
{
    SAFE( cudaFree(d_shapes) );
    SAFE( cudaFree(d_factor) );
    SAFE( cudaFree(d_raydirs) );
    SAFE( cudaFree(d_materials) );
    SAFE( cudaFree(d_random) );
    SAFE( cudaFree(d_buffer) );
    SAFE( cudaFree(d_film) );
}

void Scene::render(color_t* buffer)
{
//    start_timer();
    cudaproduceray(p_cam->gpu_type(), d_raydirs, d_factor, d_result, d_film, sample);
    cudapathtrace (p_cam->gpu_type(), d_raydirs, d_factor, d_result, d_random, d_materials, d_shapes, 8); // FIXME: hardcoded shape size
    cudargbtoint  (p_cam->gpu_type(), d_result, d_film, sample, d_buffer);

    printf("%d\n", sample++);

    SAFE( cudaMemcpy( buffer, d_buffer, p_cam->size() * sizeof(color_t), cudaMemcpyDeviceToHost) );
//    end_timer();
}
