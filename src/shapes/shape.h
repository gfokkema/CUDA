#ifndef SHAPE_H_
#define SHAPE_H_

#define EPSILON 1e-4

#include "../util/ray.h"

class Shape {
public:
	Shape(Vector origin) : _origin(origin) {};
	virtual ~Shape() {};
	virtual bool intersect(const Ray& ray, Vector& hit, Vector& normal) = 0;
protected:
	Vector _origin;
};

#endif /* SHAPE_H_ */
