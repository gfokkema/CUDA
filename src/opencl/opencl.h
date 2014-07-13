#ifndef OPENCL_H_
#define OPENCL_H_

#include <CL/cl.hpp>

struct cl_shape;
class Camera;

class OpenCL {
public:
	OpenCL();
	virtual ~OpenCL();

	int init();
	int load(std::string kernel_path);

	int produceray(Camera* cam, cl_float4*& raydirs);
	int traceray(Camera *cam, cl_float4* raydirs, std::vector<cl_shape*> shapes, unsigned char*& buffer);

	cl::CommandQueue queue;
	cl::Context context;
protected:
	cl::Device device;
	cl::Program program;
};

#endif /* OPENCL_H_ */
