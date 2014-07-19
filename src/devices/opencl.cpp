#include <fstream>
#include <iostream>
#include <sstream>

#include "opencl.h"

OpenCL::OpenCL() {}

OpenCL::~OpenCL() {}

int OpenCL::init() {
	std::vector<cl::Platform> platforms;
	SAFE(cl::Platform::get(&platforms));

	float max_ver;
	for (int i = 0; i < platforms.size(); i++) {
		std::vector<cl::Device> devices;
		SAFE(platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices));

		std::string version;
		SAFE(platforms[i].getInfo(CL_PLATFORM_VERSION, &version));

		float ver = atof(version.substr(7,3).c_str());
		if (ver > max_ver && devices.size() > 0) {
			max_ver = ver;
			device = devices[0];
		}
	}

	std::string platversion, devicename;
	SAFE(device.getInfo(CL_DEVICE_VERSION, &platversion));
	SAFE(device.getInfo(CL_DEVICE_NAME, &devicename));

	std::cout << "Selected platform:\t" << platversion << std::endl;
	std::cout << "Selected device:\t" << devicename << std::endl;

	SAFE_REF(context = cl::Context(device, 0, 0, 0, &err));
	SAFE_REF(queue = cl::CommandQueue(context, device, 0, &err));

	std::vector<std::string> source_paths;
	source_paths.push_back("../src/util/gpu_types.h");
	source_paths.push_back("../src/kernel/kernel.cl");

	SAFE(this->load(source_paths));

	return CL_SUCCESS;
}

int OpenCL::load(std::vector<std::string> source_paths) {
	cl::Program::Sources source;

	std::vector<std::string> txt_sources;
	for (std::string source_path : source_paths) {
		std::ifstream file(source_path);
		std::stringstream buffer;
		std::string strbuffer;

		buffer << file.rdbuf();
		strbuffer = buffer.str();
		const char* txt_source = strbuffer.c_str();

		source.push_back(std::make_pair(txt_source, strlen(txt_source)));
		txt_sources.push_back(strbuffer);
	}

	SAFE_REF(program = cl::Program(context, source, &err));

	std::vector<cl::Device> devices(1, device);
	SAFE_BUILD(program.build(devices, 0, 0, 0));

	return CL_SUCCESS;
}

int OpenCL::produceray(Camera* cam, float4*& raydirs) {
	unsigned size = cam->width() * cam->height();
	cl::Buffer cl_write;
	cl::Kernel kernel;

	// Initialize write buffer
	SAFE_REF(cl_write = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, size * sizeof(cl_float4), 0, &err));

	// Initialize kernel
	SAFE_REF(kernel = cl::Kernel(program, "produceray", &err));
	SAFE(kernel.setArg(0, cl_write));
	SAFE(kernel.setArg(1, cam->gpu_type()));

	// Enqueue kernel
	cl::NDRange global(cam->height());
	SAFE(this->queue.enqueueNDRangeKernel(kernel, 0, global));
	SAFE(this->queue.finish());

	// Read results
	raydirs = new float4[size];
	SAFE(this->queue.enqueueReadBuffer(cl_write, CL_TRUE, 0, size * sizeof(float4), raydirs));

	return CL_SUCCESS;
}

int OpenCL::traceray(Camera *cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer) {
	unsigned size = cam->width() * cam->height();
	cl::Buffer cl_read_rays;
	cl::Buffer cl_read_shapes;
	cl::Buffer cl_write;
	cl::Kernel kernel;

	SAFE_REF(cl_read_rays = cl::Buffer(this->context, CL_MEM_READ_ONLY, size * sizeof(float4), 0, &err));
	SAFE_REF(cl_read_shapes = cl::Buffer(this->context, CL_MEM_READ_ONLY, shapes.size() * sizeof(shape), 0, &err));
	SAFE_REF(cl_write = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, 3 * size * sizeof(unsigned char), 0, &err));

	SAFE(this->queue.enqueueWriteBuffer(cl_read_rays, CL_TRUE, 0, size * sizeof(float4), raydirs));
	SAFE(this->queue.enqueueWriteBuffer(cl_read_shapes, CL_TRUE, 0, shapes.size() * sizeof(shape), shapes.data()));

	// Initialize kernel
	SAFE_REF(kernel = cl::Kernel(program, "traceray", &err));
	SAFE(kernel.setArg(0, cam->pos().gpu_type()));
	SAFE(kernel.setArg(1, cl_read_rays));
	SAFE(kernel.setArg(2, cl_read_shapes));
	SAFE(kernel.setArg(3, cl_write));

	// Enqueue kernel
	cl::NDRange global(size);
	SAFE(this->queue.enqueueNDRangeKernel(kernel, 0, global));
	SAFE(this->queue.finish());

	// Read results
	SAFE(this->queue.enqueueReadBuffer(cl_write, CL_TRUE, 0, 3 * size * sizeof(unsigned char), buffer));

	delete raydirs;

	return CL_SUCCESS;
}
