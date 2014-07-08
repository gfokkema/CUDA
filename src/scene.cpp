#include "shapes/shape.h"
#include "shapes/sphere.h"
#include "util/ray.h"
#include "scene.h"

Scene::Scene() : cam() {
	// Initialize shapes here.
	_shapes.push_back(new Sphere(Vector(0,0,-3), .2));
}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

void Scene::render(unsigned char* buffer, int width, int height) {
	Vector up = cam.up();
	Vector right = cam.right();
	Vector pos = cam.pos();
	Vector dir = cam.dir();

	float invwidth = 1.f / width;
	float invheight = 1.f / height;

	unsigned char *channel = buffer;
	for (int yi = 0; yi < height; yi++) {
		for (int xi = 0; xi < width; xi++) {
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

const Vector Scene::trace(const Ray ray) {
	for(Shape* shape : _shapes) {
		Vector hit, normal;
		if (shape->intersect(ray, hit, normal)) {
			return Vector(1,0,0);
		}
	}
	return Vector(0,0,0);
}
