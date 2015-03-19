#include "common.h"
#include "cuda/host_kernels.cuh"

#include <chrono>
#include <iomanip>

#ifndef SCENE_H_
#define SCENE_H_

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
    unsigned char* d_buffer;
    float4*        d_random;
    float4*        d_raydirs;
    shape_t*       d_shapes;
};

#endif /* SCENE_H_ */
