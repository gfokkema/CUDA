#include <ctime>

#include "scene.h"

#include "util/camera.h"
#include "util/ray.h"

Scene::Scene() : _cam(nullptr) {
	// Initialize shapes here.
	_shapes.push_back(shape { { Vector(0,0,-3).gpu_type(), .2 }, SPHERE });
}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

void Scene::setCamera(Camera* cam) {
	this->_cam = cam;
}

void Scene::render(unsigned char* buffer) {
	std::clock_t c_start = std::clock();

	float4* gpuraydirs;
	cudaproduceray(_cam->gpu_type(), gpuraydirs);
	cudatraceray  (_cam->gpu_type(), gpuraydirs, _shapes.data(), buffer);

	std::clock_t c_end = std::clock();
	std::cout << "Frame duration:\t" << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\r";
	std::flush(std::cout);
}
