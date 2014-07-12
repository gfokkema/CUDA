#include "opencl.h"
#include "produceray.h"

#include "util/camera.h"

ProduceRay::ProduceRay(OpenCL* opencl) : _opencl(opencl) {
	opencl->load("../src/kernel/produceray.cl", kernel);
}

ProduceRay::~ProduceRay() {
	// TODO Auto-generated destructor stub
}

int ProduceRay::perform(Camera* cam, cl_float4*& raydirs) {
	cl_int err;
	unsigned size = cam->width() * cam->height();

	// Initialize write buffer
	cl::Buffer cl_write = cl::Buffer(this->_opencl->context, CL_MEM_WRITE_ONLY, size * sizeof(cl_float4));

	// Initialize arguments
	err = kernel.setArg(0, cl_write);
	if (err != CL_SUCCESS) std::cout << "arg1 error:" << err << std::endl;
	err = kernel.setArg(1, cam->cl_type());
	if (err != CL_SUCCESS) std::cout << "arg2 error: " << err << std::endl;

	// Enqueue kernel
	cl::NDRange global(cam->height());
	err = this->_opencl->queue.enqueueNDRangeKernel(kernel, 0, global);
	if (err != CL_SUCCESS) std::cout << "kernel error: " << err << std::endl;
	err = this->_opencl->queue.finish();
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	// Read results
	raydirs = new cl_float4[size];
	err = this->_opencl->queue.enqueueReadBuffer(cl_write, CL_TRUE, 0, size * sizeof(cl_float4), raydirs);
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	return CL_SUCCESS;
}
