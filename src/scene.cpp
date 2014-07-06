#include "shapes/shape.h"
#include "shapes/sphere.h"

#include "scene.h"

Scene::Scene() {
	// Initialize shapes here.
	_shapes.push_back(new Sphere(Vector(0,0,-3), .2));
}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

const Vector Scene::trace(const Ray ray) {
	for(Shape* shape : _shapes) {
		Vector hit, normal;
		if (shape->intersect(ray, hit, normal)) {
			std::cout << "hit: " << hit << std::endl;
			return Vector(1,0,0);
		}
	}
	std::cout << "miss: " << ray.dir() << std::endl;
	return Vector(0,0,0);
}
