#ifndef SPHERE_H_
#define SPHERE_H_

#include "shape.h"

class Sphere : public Shape {
public:
	Sphere(Vector origin, float radius);
	virtual ~Sphere();
	virtual bool intersect(const Ray& ray, Vector& hit, Vector& normal);
private:
	float _radius;
};

#endif /* SPHERE_H_ */
