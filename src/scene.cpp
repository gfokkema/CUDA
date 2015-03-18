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
: _cam(cam),
  d_buffer(nullptr),
  d_shapes(nullptr),
  d_raydirs(nullptr)
{
	// Initialize shapes here.
	_shapes.push_back(shape { { Vector(0,0,-3).gpu_type(), .2 }, SPHERE });

	cudamallocbuffer(this->d_buffer, _cam->size());
	cudamallocshapes(this->d_shapes, _shapes.data(), _shapes.size());
}

Scene::~Scene() {}

void Scene::render(unsigned char* buffer) {
    start_timer();
	cudaproduceray(_cam->gpu_type(), d_raydirs);
	cudatraceray  (_cam->gpu_type(), d_raydirs, d_shapes, d_buffer);
	cudareadbuffer(buffer, d_buffer, _cam->size());
	end_timer();
}
