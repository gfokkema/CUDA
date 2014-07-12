#include <ctime>

#include "shapes/shape.h"
#include "shapes/sphere.h"
#include "util/camera.h"
#include "util/ray.h"
#include "scene.h"

Scene::Scene() : _cam(nullptr) {
	// Initialize shapes here.
	_shapes.push_back(new Sphere(Vector(0,0,-3), .2));
}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

void Scene::setCamera(Camera* cam) {
	this->_cam = cam;
}

void Scene::render(unsigned char* buffer) {
	Vector up = _cam->up();
	Vector right = _cam->right();
	Vector pos = _cam->pos();
	Vector dir = _cam->dir();


	std::clock_t c_start = std::clock();

	cl_float4* gpuraydirs;
	rays.perform(_cam, gpuraydirs);

	unsigned size = _cam->width() * _cam->height();
	unsigned char *channel = buffer;
	for (unsigned i = 0; i < size; i++) {
		Ray ray(pos, gpuraydirs[i]);

		Vector pixel = this->trace(ray);
		for (int i = 0; i < 3; i++) {
			*channel++ = (unsigned char)(std::min(1.f, pixel[i]) * 255);
		}
	}
	std::clock_t c_end = std::clock();
	printf("Test duration (regular): %f ms\n", 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC);

	delete gpuraydirs;
}

const Vector Scene::trace(const Ray& ray) {
	for(Shape* shape : _shapes) {
		Vector hit, normal;
		if (shape->intersect(ray, hit, normal)) {
			return Vector(1,0,0);
		}
	}
	return Vector(0,0,0);
}
