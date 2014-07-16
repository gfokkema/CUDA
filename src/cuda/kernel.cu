#include "kernel.cuh"

#include <stdio.h>

__device__
unsigned char intersect(__const__ float4 origin, float4 dir, shape shape)
{
	float4 trans_origin = origin - shape.sphere.origin;
	float a = dir * dir;
	float b = 2 * trans_origin * dir;
	float c = trans_origin * trans_origin - shape.sphere.radius * shape.sphere.radius;

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
	unsigned size = cam.height * cam.width;

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

__global__
void traceray(__const__ float4 origin, float4* read_rays, shape* read_shapes, unsigned char* write_buffer)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	write_buffer[3 * idx] = intersect(origin, read_rays[idx], read_shapes[0]);
	write_buffer[3 * idx + 1] = 0;
	write_buffer[3 * idx + 2] = 0;
}

__host__
int cudatraceray(camera cam, float4* raydirs, shape* read_shapes, unsigned char*& buffer) {
	unsigned size = cam.height * cam.width;

	float4* d_raydirs;
	shape* d_shapes;
	cudaMalloc(&d_raydirs, size * sizeof(float4));
	cudaMemcpy(d_raydirs, raydirs, size * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMalloc(&d_shapes, sizeof(shape));
	cudaMemcpy(d_shapes, read_shapes, sizeof(shape), cudaMemcpyHostToDevice);

	unsigned char* d_buffer;
	cudaMalloc(&d_buffer, 3 * size * sizeof(unsigned char));

	// Perform computation on device
	traceray <<< cam.height,cam.width >>> (cam.pos, d_raydirs, d_shapes, d_buffer);

	// Read results
	cudaMemcpy(buffer, d_buffer, 3 * size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_raydirs);
	cudaFree(d_shapes);
	cudaFree(d_buffer);

	return 0;
}
