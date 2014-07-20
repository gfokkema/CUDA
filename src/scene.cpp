#include <ctime>

#include "scene.h"

#include "devices/device.h"
#include "util/camera.h"
#include "util/ray.h"

Scene::Scene(Device* device) : _cam(nullptr), _device(device) {
	// Initialize shapes here.
	_shapes.push_back(shape { { Vector(0,0,-3).gpu_type(), .2 }, SPHERE });
}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

void Scene::setCamera(Camera* cam) {
	this->_cam = cam;
}
