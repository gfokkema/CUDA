#ifndef DEVICE_DEVICE_H_
#define DEVICE_DEVICE_H_

#include <common.h>

class Scene;

class Device {
public:
    Device() {};
    virtual ~Device() {};

    virtual void copy(std::vector<mat_t> materials, std::vector<shape_t> shapes) = 0;

    virtual double producerays(camera_t & camera, unsigned camsize, unsigned sample) = 0;
    virtual double pathtrace(camera_t & camera) = 0;
    virtual double rgbtoint(camera_t & camera, unsigned sample) = 0;
    virtual double write(color_t * buffer, unsigned size) = 0;

    virtual double render(color_t * buffer, Scene * scene);
};



#endif /* DEVICE_DEVICE_H_ */
