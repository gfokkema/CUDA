#ifndef TRACERAY_H_
#define TRACERAY_H_

#include "opencl.h"

class TraceRay {
public:
	TraceRay(OpenCL* opencl);
	virtual ~TraceRay();
private:
	OpenCL* _opencl;
	cl::Kernel kernel;
};

#endif /* TRACERAY_H_ */
