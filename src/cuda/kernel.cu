#include "kernel.cuh"

#include <stdio.h>

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
float length(const float4& lhs) {
	return sqrt(lhs * lhs);
}

__device__
float4 normalize(const float4& lhs) {
	return lhs / length(lhs);
}

__global__
void produceray(__const__ camera cam, float4* raydirs) {
	int yi = blockIdx.x;

	unsigned offset = yi * cam.width;

	float invwidth = 1.f / cam.width;
	float invheight = 1.f / cam.height;

	for (int xi = 0; xi < cam.width; xi++) {
		float x = (xi + .5) * invwidth - 0.5;
		float y = (yi + .5) * invheight - 0.5;

		raydirs[offset + xi] = normalize(x * cam.right + y * cam.up + cam.dir);
	}
}

__host__
int cudaproduceray(camera cam, float4*& raydirs) {
	int size = cam.height * cam.width;

	float4* d_raydirs;
	cudaMalloc(&d_raydirs, size * sizeof(float4));

	// Perform computation on device
	produceray <<< cam.height,1 >>> (cam, d_raydirs);

	// Read Results
	raydirs = new float4[size];
	cudaMemcpy(raydirs, d_raydirs, size * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaFree(d_raydirs);

	return 0;
}
