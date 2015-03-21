#include "common.h"

#ifndef SCENE_H_
#define SCENE_H_

class Camera;
class Ray;

class Scene {
public:
    Scene         (Camera* cam);
    virtual ~Scene();

    void render       (color_t* buffer);
    const Vector trace(const Ray& ray);
private:
    Camera*        p_cam;
    float4*        d_accum;
    color_t*       d_buffer;
    ray_t*         d_raydirs;
    float4*        d_reflect;
    float4*        d_result;
    float4*        d_random;
    mat_t*         d_materials;
    shape_t*       d_shapes;
};

#endif /* SCENE_H_ */
