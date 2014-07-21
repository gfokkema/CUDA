#ifndef CPUDEVICE_H_
#define CPUDEVICE_H_

#define EPSILON 1e-4

#include "devices/device.h"

class CPUDevice : public Device {
public:
	CPUDevice();
	virtual ~CPUDevice();

	int init();

	int produceray(Camera* cam, gpu_float4*& raydirs);
	int traceray(Camera *cam, gpu_float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer);
private:
	unsigned char intersect(Vector origin, Vector dir, shape shape);
};

#endif /* CPUDEVICE_H_ */
