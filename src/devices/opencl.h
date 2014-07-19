#ifndef OPENCL_H_
#define OPENCL_H_

#ifdef __APPLE__
  #include <OpenCL/cl.hpp>
#else
  #include <CL/cl.hpp>
#endif

#include <vector>

#include "devices/device.h"

#define SAFE( call) {                                                     \
	cl_int err = call;                                                    \
	if( CL_SUCCESS != err) {                                              \
		fprintf(stderr, "OpenCL error in file '%s' in line %i: %d.\n",   \
				__FILE__, __LINE__, err);                                 \
				exit(EXIT_FAILURE);                                       \
	}                                                                     \
}
#define SAFE_REF( call) {                                                 \
	cl_int err; call;                                                     \
	if( CL_SUCCESS != err) {                                              \
		fprintf(stderr, "OpenCL error in file '%s' in line %i: %d.\n",   \
				__FILE__, __LINE__, err);                                 \
				exit(EXIT_FAILURE);                                       \
	}                                                                     \
}
#define SAFE_BUILD( call) {                                                   \
	cl_int err = call;                                                        \
	if( CL_SUCCESS != err) {                                                  \
		fprintf(stderr, "OpenCL error in file '%s' in line %i: %d.\n"        \
				"Error code: %d. Build log:\n%s\n",                     \
				__FILE__, __LINE__, err,                                      \
				program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device),        \
				program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());  \
				exit(EXIT_FAILURE);                                           \
	}                                                                         \
}

class OpenCL : public Device {
public:
	OpenCL();
	virtual ~OpenCL();

	int init();
	int load(std::vector<std::string> source_paths);

	int produceray(Camera* cam, float4*& raydirs);
	int traceray(Camera *cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer);

	cl::CommandQueue queue;
	cl::Context context;
protected:
	cl::Device device;
	cl::Program program;
};

#endif /* OPENCL_H_ */
