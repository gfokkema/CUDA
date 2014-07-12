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
	float invwidth = 1.f / _cam->width();
	float invheight = 1.f / _cam->height();

	unsigned size = _cam->width() * _cam->height();
	cl_float4* raydirs = new cl_float4[size];
	unsigned char *channel = buffer;
	unsigned offset = 0;
	for (int yi = 0; yi < _cam->height(); yi++) {
		offset = yi * _cam->width();
		for (int xi = 0; xi < _cam->width(); xi++) {
			float x = (xi + .5) * invwidth - 0.5;
			float y = (yi + .5) * invheight - 0.5;

			Vector raydir = (x * right + y * up + dir).normalize();
			raydirs[offset + xi] = raydir.cl_type();
			//Ray ray(pos, raydir);

			//Vector pixel = this->trace(ray);
			//for (int i = 0; i < 3; i++) {
			//	*channel++ = (unsigned char)(std::min(1.f, pixel[i]) * 255);
			//}
		}
	}
	std::clock_t c_end = std::clock();
	printf("Test duration (regular): %f ms\n", 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC);

	c_start = std::clock();
	cl_float4* gpuraydirs;
	rays.perform(_cam, gpuraydirs);
	c_end = std::clock();
	printf("Test duration (OpenCL): %f ms\n", 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC);

	unsigned count = 0;
	for (unsigned i = 0; i < size; i++) {
		bool equal = true;
		for (int j = 0; j < 3; j++) {
			if (fabs(raydirs[i].v4[j] - gpuraydirs[i].v4[j]) > 0.000001) equal = false;
		}
		if (equal) count++;
	}
	std::cout << "result: " << count << "/" << size << " correct." << std::endl;

	delete raydirs;
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
