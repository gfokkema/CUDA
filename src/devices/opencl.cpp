#include <fstream>
#include <iostream>
#include <sstream>

#include "opencl.h"

OpenCL::OpenCL() {
	this->init();
}

OpenCL::~OpenCL() {}

int OpenCL::init() {
	cl_int err;
	cl_uint num_plats;

	err = clGetPlatformIDs(0, NULL, &num_plats);
	if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);

	cl_platform_id plat[num_plats];
	err = clGetPlatformIDs(num_plats, plat, NULL);
	if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);

	float max_ver;
	for (int i = 0; i < num_plats; i++) {
		cl_uint num_devices;
		size_t plat_info_length;

		err = clGetDeviceIDs(plat[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);

		cl_device_id devices[num_devices];
		err = clGetDeviceIDs(plat[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);

		err = clGetPlatformInfo(plat[i], CL_PLATFORM_VERSION, 0, NULL, &plat_info_length);
		if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);

		char plat_version[plat_info_length];
		err = clGetPlatformInfo(plat[i], CL_PLATFORM_VERSION, plat_info_length, plat_version, NULL);
		if (err != CL_SUCCESS)		printf("ERROR at line %u\n", __LINE__);

		std::string version(plat_version);
		std::cout << version << std::endl;
		float ver = atof(version.substr(7,3).c_str());
		if (ver > max_ver && num_devices > 0) {
			max_ver = ver;
			device = devices[0];
		}
	}

	char device_name[256];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 256, device_name, NULL);
	char device_version[256];
	clGetDeviceInfo(device, CL_DEVICE_VERSION, 256, device_version, NULL);

	std::cout << "Selected platform:\t" << device_version << std::endl;
	std::cout << "Selected device:\t" << device_name << std::endl;

	// Set OpenCL context
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) return err;

	// Create command queue
	commands = clCreateCommandQueue(context, device, 0, &err);
	if (err != CL_SUCCESS) return err;

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

	std::cout << "--- BUILD SUCCESS ---" << std::endl;
	return CL_SUCCESS;
}

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

	clFinish(commands);

	return CL_SUCCESS;
}
