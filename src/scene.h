#ifndef SCENE_H_
#define SCENE_H_

#include <vector>

#include "util/vector.h"

extern "C" int cudaproduceray  (camera_t       cam,
                                float4*&       d_raydirs);
extern "C" int cudatraceray    (camera_t       cam,
                                float4*        d_raydirs,
                                shape_t*       d_shapes,
                                unsigned char* d_buffer);

extern "C" int cudamallocshapes(shape_t*&       d_shapes,
                                shape_t*        shapes,
                                int             size);
extern "C" int cudamallocbuffer(unsigned char*& d_buffer,
                                int size);
extern "C" int cudareadbuffer  (unsigned char*  buffer,
                                unsigned char*  d_buffer,
                                int size);

class Camera;
class Ray;

class Scene {
public:
	Scene         (Camera* cam);
	virtual ~Scene();

	void render(unsigned char* buffer);
	const Vector trace(const Ray& ray);
private:
	Camera*        p_cam;
	shape_list_t   p_shapes;
	unsigned char* d_buffer;
	shape_t*       d_shapes;
	float4*        d_raydirs;
};

#endif /* SCENE_H_ */
