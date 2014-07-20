#ifndef OPENCL_H_
#define OPENCL_H_

#ifdef __APPLE__
  #include <OpenCL/cl.hpp>
#else
  #include <CL/cl.hpp>
#endif

#include "devices/device.h"

class OpenCL : public Device {
public:
	OpenCL();
	virtual ~OpenCL();

	int init();
	int load(std::string kernel_path);

	int produceray(Camera* cam, float4*& raydirs);
	int traceray(Camera *cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer);

	virtual device_mem malloc(size_t size, permission perm);
	virtual void read(device_mem mem, size_t size, void* data_read);
	virtual void write(device_mem mem, size_t size, void* data_write);
	virtual int enqueue_kernel_range(kernel_key id, uint8_t num_args, void** arg_values, size_t* arg_sizes, uint8_t dim, size_t* work_size);

	cl_command_queue 	commands;
	cl_context		context;
protected:
	cl_device_id		device;
	std::vector<cl_kernel>	kernels;
};

#endif /* OPENCL_H_ */
