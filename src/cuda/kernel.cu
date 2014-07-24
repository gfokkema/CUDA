#include "kernel.cuh"

#include <stdio.h>

__device__
unsigned char intersect(__const__ float4 origin, float4 dir, shape shape)
{
	float4 trans_origin = origin - shape.data.sp.origin;
	float a = dir * dir;
	float b = 2 * trans_origin * dir;
	float c = trans_origin * trans_origin - shape.data.sp.radius * shape.data.sp.radius;

	float disc = b * b - 4 * a * c;
	if (disc < 0)	return 0;

	// We use the following in place of the quadratic formula for
	// more numeric precision.
	float q = (b > 0) ?
				-0.5 * (b + sqrt(disc)) :
				-0.5 * (b - sqrt(disc));
	float t0 = q / a;
	float t1 = c / q;
	//if (t0 < t1) swap(t0,t1);

	float t;
	if (t0 < EPSILON)	return 0;
	if (t1 < 0)		t = t0;
	else			t = t1;

	return 255;
}

__global__
void produceray(
		__const__ camera* cam,
		float4* raydirs) {
	unsigned xi = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned yi = blockIdx.y * blockDim.y + threadIdx.y;

	float x = (xi + .5) / cam->width - 0.5;
	float y = (yi + .5) / cam->height - 0.5;

	raydirs[yi * cam->width + xi] = normalize(x * cam->right + y * cam->up + cam->dir);
}

__global__
void traceray(
		__const__ camera* cam,
		__const__ float4* read_rays,
		__const__ shape* read_shapes,
		unsigned char* write_buffer)
{
	int xi = blockIdx.x * blockDim.x + threadIdx.x;
	int yi = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned idx = (yi * cam->width + xi);
	write_buffer[3 * idx] = intersect(cam->pos, read_rays[idx], read_shapes[0]);
	write_buffer[3 * idx + 1] = 0;
	write_buffer[3 * idx + 2] = 0;
}

int cudaproduceray(
		dim3 blocks, dim3 threads,
		__const__ camera* cam,
		float4* raydirs) {
	produceray <<< blocks, threads >>> (cam, raydirs);
}

int cudatraceray(
		dim3 blocks, dim3 threads,
		__const__ camera* cam,
		__const__ float4* read_rays,
		__const__ shape* read_shapes,
		unsigned char* write_buffer) {
	traceray <<< blocks, threads >>> (cam, read_rays, read_shapes, write_buffer);
}
