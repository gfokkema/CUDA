#ifndef SCENE_H_
#define SCENE_H_

#include "cuda/host_kernels.cuh"
#include "util/vector.h"

class Camera;
class Ray;

class Scene {
public:
    Scene         (Camera* cam);
    virtual ~Scene();

    void render       (unsigned char* buffer);
    const Vector trace(const Ray& ray);
private:
    Camera*        p_cam;
    shape_list_t   p_shapes;
    unsigned char* d_buffer;
    shape_t*       d_shapes;
    float4*        d_raydirs;
};

#endif /* SCENE_H_ */
