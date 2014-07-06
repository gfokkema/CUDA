#ifndef SHAPE_H_
#define SHAPE_H_

class Shape {
public:
	virtual Shape() = 0;
	virtual ~Shape() = 0;
	virtual bool intersect(Ray ray, Vector& hit, Vector& normal) = 0;
};

#endif /* SHAPE_H_ */
