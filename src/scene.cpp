#include <chrono>
#include <iomanip>

#include "scene.h"

#include "util/camera.h"
#include "util/ray.h"

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
  d_raydirs(nullptr),
  d_shapes(nullptr)
{
    // Initialize shapes here.
    std::vector<shape_t> shapes;
    shapes.push_back({ { Vector(0,0,-3).gpu_type(), .2 }, SPHERE });

    cudamallocbuffer(this->d_buffer, p_cam->size());
    //	cudamallocshapes(this->d_shapes, shape_list_t { (int)shapes.size(), shapes.data() });
    cudamallocshapes(this->d_shapes, shapes.data(), shapes.size());

}

Scene::~Scene() {}

void Scene::render(unsigned char* buffer) {
    start_timer();
    cudaproduceray(p_cam->gpu_type(), d_raydirs);
    cudatraceray  (p_cam->gpu_type(), d_raydirs, d_shapes, d_buffer);
    cudareadbuffer(buffer, d_buffer, p_cam->size());
    end_timer();
}
