#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#define EPSILON 1e-4

#ifdef __APPLE_KERNEL_COMPILE__
#define gpu_float4 float4
#else
#if !defined(__CUDACC__) && !defined(__OPENCL_VERSION__)
typedef struct gpu_float4 {
	float v4[4];
} gpu_float4 __attribute ((aligned(16)));
#else
#define gpu_float4 float4
#endif
#endif

typedef struct camera {
	gpu_float4 pos;
	gpu_float4 dir;
	gpu_float4 up;
	gpu_float4 right;
	int width, height;
	int _,__;
} camera __attribute ((aligned(16)));

typedef enum _type {
	SPHERE,
	PLANE,
	TRIANGLE
} shape_type;

// SPHERE
typedef struct _sphere{
	gpu_float4 origin;	// offset 0
	float radius;	// offset 16
} sphere;

// PLANE
typedef struct _plane{
	gpu_float4 origin;	// offset 0
	gpu_float4 normal;	// offset 16
} plane;

// TRIANGLE
typedef struct _triangle{
	gpu_float4 v1;		// offset 0
	gpu_float4 v2;		// offset 16
	gpu_float4 v3;		// offset 32
} triangle;

typedef union _shape_data {
	sphere sp;
	plane pl;
	triangle tr;
} shape_data;

typedef struct shape {
	shape_data data;
	int type;
} shape;

#endif /* GPU_TYPES_H_ */
