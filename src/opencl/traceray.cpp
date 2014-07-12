#include "traceray.h"

#include "util/camera.h"
#include "shapes/sphere.h"

TraceRay::TraceRay(OpenCL* opencl) : _opencl(opencl) {
	opencl->load("../src/kernel/traceray.cl", kernel);
}

TraceRay::~TraceRay() {
	// TODO Auto-generated destructor stub
}

int TraceRay::perform(Camera *cam, cl_float4* raydirs, std::vector<Sphere*> shapes, unsigned char*& buffer) {
	cl_int err;
	unsigned size = cam->width() * cam->height();

	cl::Buffer cl_read_rays = cl::Buffer(this->_opencl->context, CL_MEM_READ_ONLY, size * sizeof(cl_float4));
	cl::Buffer cl_read_shapes = cl::Buffer(this->_opencl->context, CL_MEM_READ_ONLY, shapes.size() * sizeof(Sphere));
	// Initialize write buffer
	cl::Buffer cl_write = cl::Buffer(this->_opencl->context, CL_MEM_WRITE_ONLY, 3 * size * sizeof(unsigned char));

	err = this->_opencl->queue.enqueueWriteBuffer(cl_read_rays, CL_TRUE, 0, size * sizeof(cl_float4), raydirs);
	if (err != CL_SUCCESS) std::cout << "cl_read_rays error: " << err << std::endl;
	err = this->_opencl->queue.enqueueWriteBuffer(cl_read_shapes, CL_TRUE, 0, shapes.size() * sizeof(Sphere), shapes.data());
	if (err != CL_SUCCESS) std::cout << "cl_read_shapes error: " << err << std::endl;

	// Initialize arguments
	err = kernel.setArg(0, cl_read_rays);
	if (err != CL_SUCCESS) std::cout << "arg1 error:" << err << std::endl;
	err = kernel.setArg(1, cl_read_shapes);
	if (err != CL_SUCCESS) std::cout << "arg2 error: " << err << std::endl;
	err = kernel.setArg(2, cl_write);
	if (err != CL_SUCCESS) std::cout << "arg3 error: " << err << std::endl;
	err = kernel.setArg(3, cam->pos().cl_type());
	if (err != CL_SUCCESS) std::cout << "arg4 error: " << err << std::endl;

	// Enqueue kernel
	cl::NDRange global(size);
	err = this->_opencl->queue.enqueueNDRangeKernel(kernel, 0, global);
	if (err != CL_SUCCESS) std::cout << "kernel error: " << err << std::endl;
	err = this->_opencl->queue.finish();
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	// Read results
	err = this->_opencl->queue.enqueueReadBuffer(cl_write, CL_TRUE, 0, 3 * size * sizeof(unsigned char), buffer);
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	return CL_SUCCESS;
}
