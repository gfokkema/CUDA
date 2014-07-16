#ifndef __KERNEL_CUH
#define __KERNEL_CUH

#include <cuda_runtime.h>

#define EPSILON 1e-4

typedef struct camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
} camera;

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

__device__
float4 operator+(const float4& lhs, const float4& rhs) {
	float4 retval;
	retval.x = lhs.x + rhs.x;
	retval.y = lhs.y + rhs.y;
	retval.z = lhs.z + rhs.z;
	return retval;
}

__device__
float4 operator-(const float4& lhs, const float4& rhs) {
	float4 retval;
	retval.x = lhs.x - rhs.x;
	retval.y = lhs.y - rhs.y;
	retval.z = lhs.z - rhs.z;
	return retval;
}

__device__
float4 operator*(const float& lhs, const float4& rhs) {
	float4 retval;
	retval.x = lhs * rhs.x;
	retval.y = lhs * rhs.y;
	retval.z = lhs * rhs.z;
	return retval;
}

__device__
float4 operator*(const float4& lhs, const float& rhs) {
	return rhs * lhs;
}

__device__
float4 operator/(const float4& lhs, const float& rhs) {
	float4 retval;
	retval.x = lhs.x / rhs;
	retval.y = lhs.y / rhs;
	retval.z = lhs.z / rhs;
	return retval;
}

__device__
float operator*(const float4& lhs, const float4& rhs) {
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__
float length(const float4& lhs) {
	return sqrt(lhs * lhs);
}

__device__
float4 normalize(const float4& lhs) {
	return lhs / length(lhs);
}

extern "C" int cudaproduceray(camera cam, float4*& raydirs);
extern "C" int cudatraceray(camera cam, float4* raydirs, shape* read_shapes, unsigned char*& buffer);

#endif
