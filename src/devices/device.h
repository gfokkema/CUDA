#ifndef DEVICE_H_
#define DEVICE_H_

#include "shapes/shape.h"
#include "util/camera.h"

class Device {
public:
	virtual ~Device() {};

	virtual int init() = 0;

	virtual int produceray(Camera* cam, cl_float4*& raydirs) = 0;
	virtual int traceray(Camera *cam, cl_float4* raydirs, std::vector<cl_shape> shapes, unsigned char*& buffer) = 0;
};

#endif /* DEVICE_H_ */
