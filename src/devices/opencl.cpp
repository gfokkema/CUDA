#include <fstream>
#include <iostream>
#include <sstream>

#include "opencl.h"

OpenCL::OpenCL() {}

OpenCL::~OpenCL() {}

int OpenCL::init() {
	cl_int err;
	std::vector<cl::Platform> platforms;
	cl::Device device_cpp;
	cl::CommandQueue commands_cpp;
	cl::Context context_cpp;

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
			device_cpp = devices[0];
		}
	}

	std::string platversion, devicename;
	err = device_cpp.getInfo(CL_DEVICE_VERSION, &platversion);
	if (err != CL_SUCCESS) return err;
	err = device_cpp.getInfo(CL_DEVICE_NAME, &devicename);
	if (err != CL_SUCCESS) return err;

	std::cout << "Selected platform:\t" << platversion << std::endl;
	std::cout << "Selected device:\t" << devicename << std::endl;

	context_cpp = cl::Context(device_cpp, 0, 0, 0, &err);
	if (err != CL_SUCCESS) return err;

	commands_cpp = cl::CommandQueue(context_cpp, device_cpp, 0, &err);
	if (err != CL_SUCCESS) return err;

	device = device_cpp();
	commands = commands_cpp();
	context = context_cpp();

	err = this->load_kernels("../src/kernel/kernel.cl");
	if (err != CL_SUCCESS) return err;

	return CL_SUCCESS;
}

int OpenCL::load_kernels(std::string kernel_path) {
	std::ifstream file(kernel_path);
	std::stringstream buffer;
	std::string strbuffer;

	buffer << file.rdbuf();
	strbuffer = buffer.str();

	std::cout << "--------------------------" << std::endl;
	std::cout << "--- source code loaded ---" << std::endl;
	std::cout << strbuffer << std::endl;
	std::cout << "--------------------------" << std::endl;

	const char* txt_source = strbuffer.c_str();
	cl_int err;

	// Create the program from source and build it.
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &txt_source, NULL, &err);
	if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		std::cout << "Error: Failed to build program executable!" << std::endl;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		std::cout << buffer << std::endl;
	}
	cl_kernel kernel_pr = clCreateKernel(program, "produceray", &err);
	if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);
	kernels.push_back(kernel_pr);
	cl_kernel kernel_tr = clCreateKernel(program, "traceray", &err);
	if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);
	kernels.push_back(kernel_tr);

	/*
	std::vector<cl::Device> devices(1, device);
	err = program.build(devices, 0, 0, 0);
	if (err != CL_SUCCESS) {
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << std::endl;
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		return err;
	}
	*/

	std::cout << "--- BUILD SUCCESS ---" << std::endl;
	return CL_SUCCESS;
}

/*
int OpenCL::produceray(Camera* cam, float4*& raydirs) {
	cl_int err;
	unsigned size = cam->width() * cam->height();

	// Initialize write buffer
	cl::Buffer cl_write = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, size * sizeof(cl_float4));

	// Initialize kernel
	cl::Kernel kernel(program, "produceray", &err);
	err = kernel.setArg(0, cl_write);
	if (err != CL_SUCCESS) std::cout << "arg1 error:" << err << std::endl;
	err = kernel.setArg(1, cam->gpu_type());
	if (err != CL_SUCCESS) std::cout << "arg2 error: " << err << std::endl;

	// Enqueue kernel
	cl::NDRange global(cam->height());
	err = this->queue.enqueueNDRangeKernel(kernel, 0, global);
	if (err != CL_SUCCESS) std::cout << "kernel error: " << err << std::endl;
	err = this->queue.finish();
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	// Read results
	raydirs = new float4[size];
	err = this->queue.enqueueReadBuffer(cl_write, CL_TRUE, 0, size * sizeof(float4), raydirs);
	if (err != CL_SUCCESS) std::cout << "finish error: " << err << std::endl;

	return CL_SUCCESS;
}

int OpenCL::traceray(Camera *cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer) {
	cl_int err;
	unsigned size = cam->width() * cam->height();

	cl::Buffer cl_read_rays = cl::Buffer(this->context, CL_MEM_READ_ONLY, size * sizeof(float4));
	cl::Buffer cl_read_shapes = cl::Buffer(this->context, CL_MEM_READ_ONLY, shapes.size() * sizeof(shape));
	// Initialize write buffer
	cl::Buffer cl_write = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, 3 * size * sizeof(unsigned char));

	err = this->queue.enqueueWriteBuffer(cl_read_rays, CL_TRUE, 0, size * sizeof(float4), raydirs);
	if (err != CL_SUCCESS) std::cout << "cl_read_rays error: " << err << std::endl;
	err = this->queue.enqueueWriteBuffer(cl_read_shapes, CL_TRUE, 0, shapes.size() * sizeof(shape), shapes.data());
	if (err != CL_SUCCESS) std::cout << "cl_read_shapes error: " << err << std::endl;

	// Initialize kernel
	cl::Kernel kernel(program, "traceray", &err);
	err = kernel.setArg(0, cam->pos().gpu_type());
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

	delete raydirs;

	return CL_SUCCESS;
}
*/

device_mem OpenCL::malloc(size_t size, permission perm) {
	cl_mem_flags cl_perm;
	cl_int err;
	switch (perm) {
		case PERM_WRITE_ONLY:
			cl_perm = CL_MEM_WRITE_ONLY;
			break;
		case PERM_READ_ONLY:
			cl_perm = CL_MEM_READ_ONLY;
			break;
		case PERM_READ_WRITE:
			cl_perm = CL_MEM_READ_WRITE;
			break;
	}
	cl_mem buff = clCreateBuffer(context, cl_perm, size, NULL, &err);
	if (err != CL_SUCCESS) std::cout << "malloc error: " << err << std::endl;

	/* FIXME: Should this point to the cl_mem object, or should it just be the
	 * cl_mem object itself (which internally is the same as _cl_mem*) ?
	 */
	// device_mem
	return {(uintptr_t)buff, sizeof(cl_mem)};
}

void OpenCL::read(device_mem mem, size_t size, void* data_read) {
	cl_mem buff = (cl_mem) mem._mem_pointer;
	cl_int err;
	err = clEnqueueReadBuffer(commands, buff, CL_TRUE, 0, size, data_read, 0, NULL, NULL);
	if (err != CL_SUCCESS) std::cout << "read error: " << err << std::endl;
}

void OpenCL::write(device_mem mem, size_t size, void* data_write) {
	cl_mem buff = (cl_mem) mem._mem_pointer;
	cl_int err;
	//std::cout << data_write << std::endl;
	err = clEnqueueWriteBuffer(commands, buff, CL_TRUE, 0, size, data_write, 0, NULL, NULL);
	if (err != CL_SUCCESS) std::cout << "write error: " << err << std::endl;
}

int OpenCL::enqueue_kernel_range(kernel_key id, uint8_t num_args, void** arg_values,
				size_t* arg_sizes, uint8_t dim, size_t* work_size) {
	if (id >= KERNEL_COUNT)	return CL_INVALID_KERNEL;
	cl_kernel kernel = kernels[id];
	cl_int err = 0;
	for (unsigned int i = 0; i < num_args; i++) {
		err |= clSetKernelArg(kernel, i, arg_sizes[i], arg_values[i]);
		std::cout << "arg size" << i << ": " << arg_sizes[i] << std::endl;
		if (err != CL_SUCCESS) {
			std::cout << "arg" << i << " error:" << err << std::endl;
			return err;
		}
	}

	// TODO: Figure out the optimal local work size
	err = clEnqueueNDRangeKernel(	commands,	// command queue
					kernel,		// kernel_id
					dim,		// work dimension
					NULL,		// global work offset
					work_size,	// global work size(s)
					NULL,		// local work size(s) (< MAX_WORK_ITEM_SIZES[n])
					0,		// number of events in wait list
					NULL,		// wait list
					NULL);		// return event

	if (err != CL_SUCCESS) {
		std::cout << "NDRange error:" << err << std::endl;
		return err;
	}

	return CL_SUCCESS;
}
