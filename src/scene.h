#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "util/ray.h"

class Camera;
class Shape;

class Scene {
public:
	Scene();
	virtual ~Scene();

	void setCamera(Camera* cam);
	void render(unsigned char* buffer);
	const Vector trace(const Ray& ray);
private:
	Camera* _cam;
	std::vector<Shape*> _shapes;
};

#endif /* SCENE_H_ */
