#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#define EPSILON 1e-4

#if !defined(__CUDACC__) && !defined(__OPENCL_VERSION__)
typedef struct float4 {
	float x, y, z, w;
} float4 __attribute ((aligned(16)));
#endif

typedef struct camera {
	float4 pos;
	float4 dir;
	float4 right;
	int width, height;
	int pad1, pad2;
} camera __attribute ((aligned(16)));

enum type {
	SPHERE,
	PLANE,
	TRIANGLE
};

typedef struct shape {
	union {
		// SPHERE
		struct {
			float4 origin;		// offset 0
			float radius;		// offset 16
			int pad_sphere11, pad_sphere12, pad_sphere13;	// offset 32
		} sphere; 
		// PLANE
		struct {
			float4 origin;		// offset 0
			float4 normal;		// offset 16
			float4 pad_plane1;	// offset 32
		} plane;
		// TRIANGLE
		struct {
			float4 v1;		// offset 0
			float4 v2;		// offset 16
			float4 v3;		// offset 32
		} triangle;
	};

	int type;				// offset 48
	int pad1, pad2, pad3;
} shape;

#endif /* GPU_TYPES_H_ */
