#ifndef CUDADEVICE_H_
#define CUDADEVICE_H_

#include "devices/cpudevice.h"

typedef cl_float4 float4;

extern "C" {
	int cudaproduceray(const cl_camera cam, float4*& raydirs);
}

class CUDADevice : public CPUDevice {
public:
	CUDADevice();
	virtual ~CUDADevice();

	int init();

	int produceray(Camera* cam, cl_float4*& raydirs);
	//int traceray(Camera* cam, cl_float4* raydirs, std::vector<cl_shape> shapes, unsigned char*& buffer);
};

#endif /* CUDADEVICE_H_ */
