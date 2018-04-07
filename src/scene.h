#ifndef SCENE_H_
#define SCENE_H_

#include <common.h>
#include <util/camera.h>
#include <vector>

class Scene
{
public:
    Scene(Camera * cam, std::vector<mat_t> materials, std::vector<shape_t> shapes);
    ~Scene();

    Camera * camera();
    std::vector<mat_t> materials();
    std::vector<shape_t> shapes();
private:
    Camera * p_cam;
    std::vector<mat_t> m_materials;
    std::vector<shape_t> m_shapes;
};

#endif /* SCENE_H_ */
