#ifndef TRACERAY_H_
#define TRACERAY_H_

#include "opencl.h"

class Camera;
class Sphere;

class TraceRay {
public:
	TraceRay(OpenCL* opencl);
	virtual ~TraceRay();
	int perform(Camera *cam, cl_float4* raydirs, std::vector<Sphere*> shapes, unsigned char*& buffer);
private:
	OpenCL* _opencl;
	cl::Kernel kernel;
};

#endif /* TRACERAY_H_ */
