#ifndef CUDADEVICE_H_
#define CUDADEVICE_H_

#include "devices/device.h"

extern "C" int cudaproduceray(camera cam, float4*& raydirs);
extern "C" int cudatraceray(camera cam, float4* raydirs, shape* read_shapes, unsigned char*& buffer);

class CUDADevice : public Device {
public:
	CUDADevice();
	virtual ~CUDADevice();

	int init();

	int produceray(Camera* cam, float4*& raydirs);
	int traceray(Camera* cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer);
};

#endif /* CUDADEVICE_H_ */
