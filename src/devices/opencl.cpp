#include <fstream>
#include <iostream>
#include <sstream>

#include "opencl.h"

OpenCL::OpenCL() {
	this->init();
}

OpenCL::~OpenCL() {}

int OpenCL::init() {
	cl_uint num_plats;
	SAFE(clGetPlatformIDs(0, NULL, &num_plats));

	cl_platform_id plat[num_plats];
	SAFE(clGetPlatformIDs(num_plats, plat, NULL));

	float max_ver;
	for (int i = 0; i < num_plats; i++) {
		cl_uint num_devices;
		SAFE(clGetDeviceIDs(plat[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
		cl_device_id devices[num_devices];
		SAFE(clGetDeviceIDs(plat[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));

		size_t plat_info_length;
		SAFE(clGetPlatformInfo(plat[i], CL_PLATFORM_VERSION, 0, NULL, &plat_info_length));
		char plat_version[plat_info_length];
		SAFE(clGetPlatformInfo(plat[i], CL_PLATFORM_VERSION, plat_info_length, plat_version, NULL));

		std::string version(plat_version);
		std::cout << version << std::endl;
		float ver = atof(version.substr(7,3).c_str());
		if (ver > max_ver && num_devices > 0) {
			max_ver = ver;
			device = devices[0];
		}
	}

	char device_name[256];
	char device_version[256];
	SAFE(clGetDeviceInfo(device, CL_DEVICE_NAME, 256, device_name, NULL));
	SAFE(clGetDeviceInfo(device, CL_DEVICE_VERSION, 256, device_version, NULL));

	std::cout << "Selected platform:\t" << device_version << std::endl;
	std::cout << "Selected device:\t" << device_name << std::endl;

	SAFE_REF(context = clCreateContext(0, 1, &device, NULL, NULL, &err));
	SAFE_REF(commands = clCreateCommandQueue(context, device, 0, &err));

	std::vector<std::string> source_paths;
	source_paths.push_back("../src/util/gpu_types.h");
	source_paths.push_back("../src/kernel/kernel.cl");
	SAFE(this->load_kernels(source_paths));
	return CL_SUCCESS;
}

int OpenCL::load_kernels(std::vector<std::string> source_paths) {
	std::vector<std::string> sources(source_paths.size());
	const char* source_ptr[source_paths.size()];
	for (int i = 0; i < source_paths.size(); i++) {
		std::ifstream file(source_paths[i]);
		std::stringstream buffer;

		buffer << file.rdbuf();
		sources[i] = buffer.str();
		source_ptr[i] = sources[i].c_str();

		std::cout << "--------------------------" << std::endl;
		std::cout << "--- source code loaded ---" << std::endl;
		std::cout << sources[i] << std::endl;
		std::cout << "--------------------------" << std::endl;
	}

	// Create the program from source and build it.
	cl_program program;
	SAFE_REF(program = clCreateProgramWithSource(context, source_paths.size(), source_ptr, NULL, &err));
	SAFE_BUILD(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

	cl_kernel kernel_pr, kernel_tr;
	SAFE_REF(kernel_pr = clCreateKernel(program, "produceray", &err));
	SAFE_REF(kernel_tr = clCreateKernel(program, "traceray", &err));
	kernels.push_back(kernel_pr);
	kernels.push_back(kernel_tr);

	return CL_SUCCESS;
}

device_mem OpenCL::malloc(size_t size, permission perm) {
	cl_mem_flags cl_perm;

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

	cl_mem buff;
	SAFE_REF(buff = clCreateBuffer(context, cl_perm, size, NULL, &err));

	/* FIXME: Should this point to the cl_mem object, or should it just be the
	 * cl_mem object itself (which internally is the same as _cl_mem*) ?
	 */
	// device_mem
	return {(uintptr_t)buff, sizeof(cl_mem)};
}

void OpenCL::read(device_mem mem, size_t size, void* data_read) {
	cl_mem buff = (cl_mem) mem._mem_pointer;
	SAFE(clEnqueueReadBuffer(commands, buff, CL_TRUE, 0, size, data_read, 0, NULL, NULL));
}

void OpenCL::write(device_mem mem, size_t size, void* data_write) {
	cl_mem buff = (cl_mem) mem._mem_pointer;
	SAFE(clEnqueueWriteBuffer(commands, buff, CL_TRUE, 0, size, data_write, 0, NULL, NULL));
}

int OpenCL::enqueue_kernel_range(kernel_key id, uint8_t num_args, void** arg_values,
				size_t* arg_sizes, uint8_t dim, size_t* work_size) {
	if (id >= KERNEL_COUNT)	return CL_INVALID_KERNEL;

	cl_kernel kernel = kernels[id];
	for (unsigned int i = 0; i < num_args; i++) {
		SAFE(clSetKernelArg(kernel, i, arg_sizes[i], arg_values[i]));
	}

	// TODO: Figure out the optimal local work size
	SAFE(clEnqueueNDRangeKernel(	commands,	// command queue
					kernel,		// kernel_id
					dim,		// work dimension
					NULL,		// global work offset
					work_size,	// global work size(s)
					NULL,		// local work size(s) (< MAX_WORK_ITEM_SIZES[n])
					0,		// number of events in wait list
					NULL,		// wait list
					NULL));		// return event
	SAFE(clFinish(commands));

	return CL_SUCCESS;
}
