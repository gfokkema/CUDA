#ifndef CUDADEVICE_H_
#define CUDADEVICE_H_

#include "devices/cpudevice.h"

extern "C" int cudaproduceray(camera cam, float4*& raydirs);

class CUDADevice : public CPUDevice {
public:
	CUDADevice();
	virtual ~CUDADevice();

	int init();

	int produceray(Camera* cam, float4*& raydirs);
	//int traceray(Camera* cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer);
};

#endif /* CUDADEVICE_H_ */
