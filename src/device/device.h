#include "common.h"

#ifndef DEVICE_DEVICE_H_
#define DEVICE_DEVICE_H_

class Scene;

class Device {
public:
    virtual ~Device() {};

    virtual double render(color_t * buffer, Scene * scene) = 0;
};



#endif /* DEVICE_DEVICE_H_ */
