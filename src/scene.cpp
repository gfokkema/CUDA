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

	float invwidth = 1.f / _cam->width();
	float invheight = 1.f / _cam->height();

	unsigned char *channel = buffer;
	for (int yi = 0; yi < _cam->height(); yi++) {
		for (int xi = 0; xi < _cam->width(); xi++) {
			float x = (xi + .5) * invwidth - 0.5;
			float y = (yi + .5) * invheight - 0.5;

			Vector raydir = (x * right + y * up + dir).normalize();
			Ray ray(pos, raydir);

			Vector pixel = this->trace(ray);
			for (int i = 0; i < 3; i++) {
				*channel++ = (unsigned char)(std::min(1.f, pixel[i]) * 255);
			}
		}
	}
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
