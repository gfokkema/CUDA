#ifndef DEVICE_DEVICE_H_
#define DEVICE_DEVICE_H_

#include <common.h>

class Scene;

class Device {
public:
    Device() {};
    virtual ~Device() {};

    virtual void copy(std::vector<mat_t> materials, std::vector<shape_t> shapes) = 0;
    virtual double write(color_t * buffer, unsigned size) = 0;

    virtual double producerays(scene_t& scene, unsigned camsize, unsigned sample) = 0;
    virtual double pathtrace(scene_t& scene) = 0;
    virtual double rgbtoint(scene_t& scene, unsigned sample) = 0;

    virtual double render(Scene& scene, color_t* buffer);
private:
    virtual scene_t d_scene(Scene& scene) = 0;
};



#endif /* DEVICE_DEVICE_H_ */
