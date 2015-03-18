#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "util/vector.h"

extern "C" int cudaproduceray(camera cam, float4*& raydirs);
extern "C" int cudatraceray(camera cam, float4* raydirs, shape* read_shapes, unsigned char*& buffer);

struct cl_shape;
class Camera;
class Device;
class Ray;

class Scene {
public:
	Scene();
	virtual ~Scene();

	void setCamera(Camera* cam);
	void render(unsigned char* buffer);
	const Vector trace(const Ray& ray);
private:
	Camera* _cam;
	Device* _device;
	std::vector<shape> _shapes;
};

#endif /* SCENE_H_ */
