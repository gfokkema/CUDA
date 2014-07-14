#define EPSILON 1e-4

typedef struct cl_camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
} cl_camera;

typedef struct cl_shape {
	union {
		// SPHERE
		struct {
			float4 origin;	// offset 0
			float radius;	// offset 16
		} sphere;
		// PLANE
		struct {
			float4 origin;	// offset 0
			float4 normal;	// offset 16
		} plane;
		// TRIANGLE
		struct {
			float4 v1;		// offset 0
			float4 v2;		// offset 16
			float4 v3;		// offset 32
		} triangle;
	};

	int type;					// offset 48
} cl_shape;

typedef float4 cl_float4;

__global__
void produceray(__const__ cl_camera cam, cl_float4* raydirs) {
}

int cudaproduceray(const cl_camera cam, float4*& raydirs) {
  int size = cam.height * cam.width;

  float4* d_raydirs;
  cudaMalloc(&d_raydirs, size * sizeof(float4));

  // Perform computation on device
  produceray <<< size,1 >>> (cam, d_raydirs);

  // Read Results
  raydirs = new float4[size];
  cudaMemcpy(&raydirs, d_raydirs, size * sizeof(float4), cudaMemcpyDeviceToHost);
  cudaFree(d_raydirs);

  return 0;
}
