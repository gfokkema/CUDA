#ifndef __KERNEL_CUH
#define __KERNEL_CUH

#include <cuda_runtime.h>
#include "../util/gpu_types.h"

#define EPSILON 1e-4

#define SAFE( call) {                                                   \
	cudaError err = call;                                               \
	if( cudaSuccess != err) {                                           \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",   \
				__FILE__, __LINE__, cudaGetErrorString( err) );         \
				exit(EXIT_FAILURE);                                     \
	}                                                                   \
}
#define CHECK_ERROR(errorMessage) {                                             \
	cudaError_t err = cudaGetLastError();                                       \
	if( cudaSuccess != err) {                                                   \
		fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",       \
				errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );   \
				exit(EXIT_FAILURE);                                             \
	}                                                                           \
}

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

extern int cudaproduceray(dim3 blocks, dim3 threads, __const__ camera cam, float4* raydirs);
extern int cudatraceray(dim3 blocks, dim3 threads, __const__ camera cam, float4* read_rays, shape* read_shapes, unsigned char* write_buffer);

#endif
