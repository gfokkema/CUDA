#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "opencl/produceray.h"
#include "opencl/traceray.h"

class Camera;
class OpenCL;
class Ray;
class Shape;

class Scene {
public:
	Scene(OpenCL* opencl);
	virtual ~Scene();

	void setCamera(Camera* cam);
	void render(unsigned char* buffer);
	const Vector trace(const Ray& ray);
private:
	Camera* _cam;
	std::vector<Shape*> _shapes;
	ProduceRay rays;
	TraceRay tracer;
};

#endif /* SCENE_H_ */
