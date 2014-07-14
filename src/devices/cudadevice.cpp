#include <devices/cudadevice.h>

CUDADevice::CUDADevice() {}

CUDADevice::~CUDADevice() {}

int CUDADevice::init() {
	return CL_SUCCESS;
}

int CUDADevice::produceray(Camera* cam, cl_float4*& raydirs) {
	return cudaproduceray(cam->cl_type(), raydirs);
}
