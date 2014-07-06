#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "util/camera.h"
#include "util/ray.h"

class Shape;

class Scene {
public:
	Scene();
	virtual ~Scene();

	void render(Vector*& buffer, int width, int height);
	const Vector trace(const Ray ray);
private:
	Camera cam;
	std::vector<Shape*> _shapes;
};

#endif /* SCENE_H_ */
