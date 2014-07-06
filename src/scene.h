#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "util/ray.h"
#include "util/vector.h"

class Shape;

class Scene {
public:
	Scene();
	virtual ~Scene();

	const Vector trace(const Ray ray);
private:
	std::vector<Shape*> _shapes;
};

#endif /* SCENE_H_ */
