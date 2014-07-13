#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "util/vector.h"

struct cl_shape;
class Camera;
class OpenCL;
class Ray;

class Scene {
public:
	Scene(OpenCL* opencl);
	virtual ~Scene();

	void setCamera(Camera* cam);
	void render(unsigned char* buffer);
	const Vector trace(const Ray& ray);
private:
	Camera* _cam;
	OpenCL* _device;
	std::vector<cl_shape*> _shapes;
};

#endif /* SCENE_H_ */
