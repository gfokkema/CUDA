#ifndef OPENCL_H_
#define OPENCL_H_

#include <CL/cl.hpp>

class OpenCL {
public:
	OpenCL();
	virtual ~OpenCL();

	int init();
	int load(std::string kernel_path, cl::Kernel& kernel);

	cl::CommandQueue queue;
	cl::Context context;
protected:
	cl::Device device;
};

#endif /* OPENCL_H_ */
