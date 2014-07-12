#ifndef PRODUCERAY_H_
#define PRODUCERAY_H_

class Camera;
class OpenCL;

namespace cl {
class Kernel;
}

class ProduceRay {
public:
	ProduceRay(OpenCL* opencl);
	virtual ~ProduceRay();
	int perform(Camera* cam, cl_float4*& raydirs);
private:
	OpenCL* _opencl;
	cl::Kernel kernel;
};

#endif /* PRODUCERAY_H_ */
