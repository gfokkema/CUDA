#include "scene.h"

Scene::Scene(Camera& cam, std::vector<mat_t> materials, std::vector<shape_t> shapes)
: p_cam(cam),
  m_materials(materials),
  m_shapes(shapes)
{
}

Scene::~Scene()
{
}

Camera&
Scene::camera()
{
    return p_cam;
}

std::vector<mat_t>
Scene::materials()
{
    return m_materials;
}

std::vector<shape_t>
Scene::shapes()
{
    return m_shapes;
}
