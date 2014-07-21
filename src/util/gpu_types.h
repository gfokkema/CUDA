#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#define EPSILON 1e-4

#ifndef __CUDACC__
typedef struct gpu_float4 {
	float v4[4];
} gpu_float4 __attribute ((aligned(16)));
#else
typedef float4 gpu_float4;
#endif

typedef struct camera {
	int width, height;
	gpu_float4 pos;
	gpu_float4 dir;
	gpu_float4 up;
	gpu_float4 right;
} camera;

enum type {
	SPHERE,
	PLANE,
	TRIANGLE
};

typedef struct shape {
	union {
		// SPHERE
		struct {
			gpu_float4 origin;	// offset 0
			float radius;	// offset 16
		} sphere;
		// PLANE
		struct {
			gpu_float4 origin;	// offset 0
			gpu_float4 normal;	// offset 16
		} plane;
		// TRIANGLE
		struct {
			gpu_float4 v1;		// offset 0
			gpu_float4 v2;		// offset 16
			gpu_float4 v3;		// offset 32
		} triangle;
	};

	int type;					// offset 48
} shape;

#endif /* GPU_TYPES_H_ */
