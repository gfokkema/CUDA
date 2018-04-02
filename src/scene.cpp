#include "scene.h"

std::chrono::time_point<std::chrono::system_clock> start;

static int sample;

void start_timer()
{
    start = std::chrono::system_clock::now();
}

void end_timer()
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> delta_time = end - start;
    std::cout << "\e[7mFrame duration:\t" << std::setw(5) << delta_time.count()
              << " ms" << "\tFramerate:\t" << std::setw(5)
              << 1000 / delta_time.count() << " fps\r";
    std::flush(std::cout);
}

Scene::Scene(Camera* cam)
: p_cam(cam),
  d_factor(nullptr),
  d_random(nullptr),
  d_raydirs(nullptr),
  d_film(nullptr),
  d_output(nullptr),
  d_mats(nullptr),
  d_shapes(nullptr)
{
    // mat_t { { r, g, b }, emit, n, type }
    std::vector<mat_t> materials;
    materials.push_back( { { 0, 0, 0 }, { 1, 1, 1 }, DIFFUSE }); // 0: WHITE LIGHTING
    materials.push_back( { { 0, 0, 0 }, { 0, 0, 0 }, MIRROR }); // 1: REFLECTIVE
    materials.push_back( { { .6, .6, .6 }, { 0, 0, 0 }, DIFFUSE }); // 2: DIFFUSE WHITE
    materials.push_back( { { .2, .2, 1 }, { 0, 0, 0 }, DIFFUSE }); // 3: DIFFUSE BLUE
    materials.push_back( { { 1, .2, .2 }, { 0, 0, 0 }, DIFFUSE }); // 4: DIFFUSE RED
    materials.push_back( { { .2, 1, .2 }, { 0, 0, 0 }, DIFFUSE }); // 5: DIFFUSE GREEN

    // Initialize shapes here.
    // shape_t { { origin, radius }, matidx, type }
    float e = 1e5 + 1;
    std::vector<shape_t> shapes;
    shapes.push_back( { { Vector(-.5, 0, -1.5).gpu_type(), .3 }, 0, SPHERE });
    shapes.push_back( { { Vector(+.5, 0, -1.5).gpu_type(), .3 }, 1, SPHERE });
    shapes.push_back( { { Vector(0, +e, 0).gpu_type(), 1e5 }, 2, SPHERE }); // CEILING: WHITE
    shapes.push_back( { { Vector(0, -e, 0).gpu_type(), 1e5 }, 2, SPHERE }); // FLOOR: WHITE
    shapes.push_back( { { Vector(+e, 0, 0).gpu_type(), 1e5 }, 3, SPHERE }); // RIGHT: BLUE
    shapes.push_back( { { Vector(-e, 0, 0).gpu_type(), 1e5 }, 4, SPHERE }); // LEFT:  RED
    shapes.push_back( { { Vector(0, 0, -e - 1).gpu_type(), 1e5 }, 2, SPHERE }); // BACK: GREEN
    shapes.push_back( { { Vector(0, 0, +e - 1).gpu_type(), 1e5 }, 2, SPHERE }); // BEHIND: WHITE

    // PER PIXEL BUFFERS
    SAFE(cudaMalloc(&d_output, p_cam->size() * sizeof(color_t)));
    SAFE(cudaMalloc(&d_random, p_cam->size() * sizeof(float4)));
    SAFE(cudaMalloc(&d_raydirs, p_cam->size() * sizeof(ray_t)));
    SAFE(cudaMalloc(&d_factor, p_cam->size() * sizeof(float4)));
    SAFE(cudaMalloc(&d_result, p_cam->size() * sizeof(float4)));
    SAFE(cudaMalloc(&d_film, p_cam->size() * sizeof(float4)));

    // SCENE DESCRIPTION
    int matsize = materials.size() * sizeof(mat_t);
    int shapesize = shapes.size() * sizeof(shape_t);
    SAFE(cudaMalloc(&d_mats, matsize));
    SAFE(cudaMalloc(&d_shapes, shapesize));
    SAFE(cudaMemcpy(d_mats, materials.data(), matsize, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_shapes, shapes.data(), shapesize, cudaMemcpyHostToDevice));
}

Scene::~Scene()
{
    SAFE(cudaFree(d_factor));
    SAFE(cudaFree(d_raydirs));
    SAFE(cudaFree(d_random));
    SAFE(cudaFree(d_film));
    SAFE(cudaFree(d_output));
    SAFE(cudaFree(d_mats));
    SAFE(cudaFree(d_shapes));
}

scene_t Scene::gpu_type()
{
    return
    {   8, 0, p_cam->gpu_type(), d_shapes, d_mats}; // FIXME: hardcoded shape size
}

void Scene::render(color_t* output)
{
//    start_timer();
    cudaproduceray(p_cam->gpu_type(), d_raydirs, d_factor, d_result, d_film, sample);
    cudapathtrace(this->gpu_type(), d_raydirs, d_factor, d_result, d_random);
    cudargbtoint(p_cam->gpu_type(), d_result, d_film, sample, d_output);

    printf("%d\n", sample++);

    int camsize = p_cam->size() * sizeof(color_t);
    SAFE(cudaMemcpy(output, d_output, camsize, cudaMemcpyDeviceToHost));
//    end_timer();
}
