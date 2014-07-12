#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "produceray.h"

class Camera;
class Ray;
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
	ProduceRay rays;
};

#endif /* SCENE_H_ */
