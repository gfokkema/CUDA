#include <fstream>
#include <iostream>
#include <sstream>

#include "opencl.h"

OpenCL::OpenCL() {}

OpenCL::~OpenCL() {}

int OpenCL::init() {
	cl_int err;
	std::vector<cl::Platform> platforms;

	err = cl::Platform::get(&platforms);
	if (err != CL_SUCCESS) return err;

	float max_ver;
	for (int i = 0; i < platforms.size(); i++) {
		std::vector<cl::Device> devices;
		err = platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if (err != CL_SUCCESS) continue;

		std::string version;
		err = platforms[i].getInfo(CL_PLATFORM_VERSION, &version);
		if (err != CL_SUCCESS) continue;

		float ver = atof(version.substr(7,3).c_str());
		if (ver > max_ver && devices.size() > 0) {
			max_ver = ver;
			device = devices[0];
		}
	}

	std::string platversion, devicename;
	err = device.getInfo(CL_DEVICE_VERSION, &platversion);
	if (err != CL_SUCCESS) return err;
	err = device.getInfo(CL_DEVICE_NAME, &devicename);
	if (err != CL_SUCCESS) return err;

	std::cout << "Selected platform:\t" << platversion << std::endl;
	std::cout << "Selected device:\t" << devicename << std::endl;

	context = cl::Context(device, 0, 0, 0, &err);
	if (err != CL_SUCCESS) return err;

	queue = cl::CommandQueue(context, device, 0, &err);
	if (err != CL_SUCCESS) return err;

	err = this->load("../src/kernel/kernel.cl");
	if (err != CL_SUCCESS) return err;

	return CL_SUCCESS;
}

int OpenCL::load(std::string kernel_path) {
	std::ifstream file(kernel_path);
	std::stringstream buffer;
	std::string strbuffer;

	buffer << file.rdbuf();
	strbuffer = buffer.str();
	const char* txt_source = strbuffer.c_str();

	cl::Program::Sources source(1, std::make_pair(txt_source, strlen(txt_source)));
	program = cl::Program(context, source);

	std::cout << "--------------------------" << std::endl;
	std::cout << "--- source code loaded ---" << std::endl;
	std::cout << source[0].first << std::endl;
	std::cout << "--------------------------" << std::endl;

	cl_int err;
	std::vector<cl::Device> devices(1, device);
	err = program.build(devices, 0, 0, 0);
	if (err != CL_SUCCESS) {
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << std::endl;
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		return err;
	}

	std::cout << "--- BUILD SUCCESS ---" << std::endl;
	return CL_SUCCESS;
}

int OpenCL::produceray(Camera* cam, cl_float4*& raydirs) {
	cl_int err;
	unsigned size = cam->width() * cam->height();

	// Initialize write buffer
	cl::Buffer cl_write = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, size * sizeof(cl_float4));

	// Initialize kernel
	cl::Kernel kernel(program, "produceray", &err);
	err = kernel.setArg(0, cl_write);
	if (err != CL_SUCCESS) std::cout << "arg1 error:" << err << std::endl;
	err = kernel.setArg(1, cam->cl_type());
	if (err != CL_SUCCESS) std::cout << "arg2 error: " << err << std::endl;

	// Enqueue kernel
	cl::NDRange global(cam->height());
	err = this->queue.enqueueNDRangeKernel(kernel, 0, global);
	if (err != CL_SUCCESS) std::cout << "kernel error: " << err << std::endl;
	err = this->queue.finish();
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	// Read results
	raydirs = new cl_float4[size];
	err = this->queue.enqueueReadBuffer(cl_write, CL_TRUE, 0, size * sizeof(cl_float4), raydirs);
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	return CL_SUCCESS;
}

int OpenCL::traceray(Camera *cam, cl_float4* raydirs, std::vector<cl_shape> shapes, unsigned char*& buffer) {
	cl_int err;
	unsigned size = cam->width() * cam->height();

	cl::Buffer cl_read_rays = cl::Buffer(this->context, CL_MEM_READ_ONLY, size * sizeof(cl_float4));
	cl::Buffer cl_read_shapes = cl::Buffer(this->context, CL_MEM_READ_ONLY, shapes.size() * sizeof(cl_shape));
	// Initialize write buffer
	cl::Buffer cl_write = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, 3 * size * sizeof(unsigned char));

	err = this->queue.enqueueWriteBuffer(cl_read_rays, CL_TRUE, 0, size * sizeof(cl_float4), raydirs);
	if (err != CL_SUCCESS) std::cout << "cl_read_rays error: " << err << std::endl;
	err = this->queue.enqueueWriteBuffer(cl_read_shapes, CL_TRUE, 0, shapes.size() * sizeof(cl_shape), shapes.data());
	if (err != CL_SUCCESS) std::cout << "cl_read_shapes error: " << err << std::endl;

	// Initialize kernel
	cl::Kernel kernel(program, "traceray", &err);
	err = kernel.setArg(0, cam->pos().cl_type());
	if (err != CL_SUCCESS) std::cout << "arg4 error: " << err << std::endl;
	err = kernel.setArg(1, cl_read_rays);
	if (err != CL_SUCCESS) std::cout << "arg1 error:" << err << std::endl;
	err = kernel.setArg(2, cl_read_shapes);
	if (err != CL_SUCCESS) std::cout << "arg2 error: " << err << std::endl;
	err = kernel.setArg(3, cl_write);
	if (err != CL_SUCCESS) std::cout << "arg3 error: " << err << std::endl;

	// Enqueue kernel
	cl::NDRange global(size);
	err = this->queue.enqueueNDRangeKernel(kernel, 0, global);
	if (err != CL_SUCCESS) std::cout << "kernel error: " << err << std::endl;
	err = this->queue.finish();
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	// Read results
	err = this->queue.enqueueReadBuffer(cl_write, CL_TRUE, 0, 3 * size * sizeof(unsigned char), buffer);
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	return CL_SUCCESS;
}
