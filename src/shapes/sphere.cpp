#include "sphere.h"

Sphere::Sphere(Vector origin, float radius)
: Shape(origin), _radius(radius) {
}

Sphere::~Sphere() {}

bool Sphere::intersect(const Ray& ray, Vector& hit, Vector& normal) {
	Vector trans_origin = ray.pos() - this->_origin;
	float a = ray.dir() * ray.dir();
	float b = 2 * trans_origin * ray.dir();
	float c = trans_origin * trans_origin - this->_radius * this->_radius;

	float disc = b * b - 4 * a * c;
	if (disc < 0)	return false;

	// We use the following in place of the quadratic formula for
	// more numeric precision.
	float q = (b > 0) ?
				-0.5 * (b + sqrtf(disc)) :
				-0.5 * (b - sqrtf(disc));
	float t0 = q / a;
	float t1 = c / q;
	if (t0 < t1) std::swap(t0,t1);

	float t;
	if (t0 < EPSILON)	return false;
	if (t1 < 0)		t = t0;
	else			t = t1;

	normal = (trans_origin + t * ray.dir()).normalize();
	hit = ray.pos() + t * ray.dir();
	return true;
}
