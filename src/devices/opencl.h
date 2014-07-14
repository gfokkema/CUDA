#ifndef OPENCL_H_
#define OPENCL_H_

#include <CL/cl.hpp>

#include "devices/device.h"

class OpenCL : public Device {
public:
	OpenCL();
	virtual ~OpenCL();

	int init();
	int load(std::string kernel_path);

	int produceray(Camera* cam, float4*& raydirs);
	int traceray(Camera *cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer);

	cl::CommandQueue queue;
	cl::Context context;
protected:
	cl::Device device;
	cl::Program program;
};

#endif /* OPENCL_H_ */
