#include <devices/cudadevice.h>

CUDADevice::CUDADevice() {}

CUDADevice::~CUDADevice() {}

int CUDADevice::init() {
	return 0;
}

int CUDADevice::produceray(Camera* cam, float4*& raydirs) {
	return cudaproduceray(cam->gpu_type(), raydirs);
}

int CUDADevice::traceray(Camera* cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer) {
	return cudatraceray(cam->gpu_type(), raydirs, shapes.data(), buffer);
}
