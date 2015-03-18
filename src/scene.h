#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "util/vector.h"

extern "C" int cudamallocshapes(shape*& d_shapes, shape* shapes, int size);
extern "C" int cudaproduceray  (camera cam, float4*& raydirs);
extern "C" int cudatraceray    (camera cam, float4* raydirs, shape* read_shapes, unsigned char* d_buffer);
extern "C" int cudamallocbuffer(unsigned char*& d_buffer, int size);
extern "C" int cudareadbuffer  (unsigned char* buffer, unsigned char* d_buffer, int size);

class Camera;
class Device;
class Ray;

class Scene {
public:
	Scene         (Camera* cam);
	virtual ~Scene();

	void render(unsigned char* buffer);
	const Vector trace(const Ray& ray);
private:
	Camera* _cam;
	Device* _device;
	std::vector<shape> _shapes;
	unsigned char*    d_buffer;
	shape*            d_shapes;
	float4*           d_raydirs;
};

#endif /* SCENE_H_ */
