#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#define EPSILON 1e-4

#if !defined(__CUDACC__) && !defined(__OPENCL_VERSION__)
typedef struct float4 {
	float v4[4];
} float4 __attribute ((aligned(16)));
#endif /* __CUDACC__ */

typedef struct camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
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
} shape;

#endif /* GPU_TYPES_H_ */
