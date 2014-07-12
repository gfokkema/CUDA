#include "produceray.h"

#include "util/camera.h"

ProduceRay::ProduceRay() {
	if (!this->initialized) this->init();
	this->load("../src/kernel/produceray.cl", kernel);
}

ProduceRay::~ProduceRay() {
	// TODO Auto-generated destructor stub
}

int ProduceRay::perform(Camera* cam) {
	std::cout << "perform cycle" << std::endl;

	unsigned size = cam->width() * cam->height();
	float* write = new float[size];
	cl::Buffer cl_write;

	cl_int err;
	cl_write = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, size, 0, &err);
	if (err != CL_SUCCESS) return err;

	err = queue.enqueueWriteBuffer(cl_write, CL_TRUE, 0, size, write, 0, 0);
	if (err != CL_SUCCESS) std::cout << "buffer error" << std::endl;

	err = kernel.setArg(0, cl_write);
	if (err != CL_SUCCESS) std::cout << "arg1 error:" << err << std::endl;
	err = kernel.setArg(1, cam->cl_type());
	if (err != CL_SUCCESS) std::cout << "arg2 error: " << err << std::endl;

	cl::NDRange global(cam->width() * cam->height());
	queue.enqueueNDRangeKernel(kernel, 0, size);
	queue.finish();

	return CL_SUCCESS;
}
