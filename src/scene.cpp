#include <ctime>

#include "scene.h"

#include "devices/device.h"
#include "util/camera.h"
#include "util/ray.h"

Scene::Scene(Camera* cam) : _cam(cam) {
	// Initialize shapes here.
	_shapes.push_back(shape { {.pl = { Vector(0,-1,0).gpu_type(), Vector(0,1,0).gpu_type() }}, PLANE });
	_shapes.push_back(shape { {.sp = { Vector(0,0,0).gpu_type(), 1 }}, SPHERE });
}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

//void Scene::setCamera(Camera* cam) {
//	this->_cam = cam;
//}
