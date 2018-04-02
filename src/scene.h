#include "common.h"

#ifndef SCENE_H_
#define SCENE_H_

class Camera;
class Ray;

class Scene
{
public:
    Scene(Camera* cam);
    ~Scene();

    Camera* cam()
    {
        return p_cam;
    }

    scene_t gpu_type();
    void render(color_t* output);
    Vector trace(const Ray& ray);
private:
    Camera* p_cam;

    // Device buffers, these actually don't belong.
    float4* d_factor;
    ray_t* d_raydirs;
    float4* d_random;
    float4* d_result;
    float4* d_film;
    color_t* d_output;    // output color
    mat_t* d_mats;
    shape_t* d_shapes;
};

#endif /* SCENE_H_ */
