#include "kernel.cuh"

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
	unsigned xi = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned yi = blockIdx.y * blockDim.y + threadIdx.y;

	float x = (xi + .5) / cam.width - 0.5;
	float y = (yi + .5) / cam.height - 0.5;

	raydirs[yi * cam.width + xi] = normalize(x * cam.right + y * cam.up + cam.dir);
}

__global__
void traceray(__const__ camera cam, float4* read_rays, shape* read_shapes, unsigned char* write_buffer)
{
	int xi = blockIdx.x * blockDim.x + threadIdx.x;
	int yi = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned idx = (yi * cam.width + xi);
	write_buffer[3 * idx] = intersect(cam.pos, read_rays[idx], read_shapes[0]);
	write_buffer[3 * idx + 1] = 0;
	write_buffer[3 * idx + 2] = 0;
}

__host__
int cudaproduceray(camera cam, float4*& d_raydirs) {
	unsigned size = cam.height * cam.width;

	SAFE(cudaMalloc(&d_raydirs, size * sizeof(float4)));

	// Perform computation on device
	dim3 threadsperblock(8, 8);
	dim3 numblocks(	cam.width / threadsperblock.x,
					cam.height / threadsperblock.y);
	produceray <<< numblocks,threadsperblock >>> (cam, d_raydirs);
	CHECK_ERROR("Launching produce kernel");

	return 0;
}

__host__
int cudatraceray(camera cam,
                 float4* d_raydirs,
                 shape*  d_shapes,
                 unsigned char* d_buffer)
{	
	// Perform computation on device
	dim3 threadsperblock(8, 8);
	dim3 numblocks(	cam.width / threadsperblock.x,
					cam.height / threadsperblock.y);
	traceray <<< numblocks,threadsperblock >>> (cam, d_raydirs, d_shapes, d_buffer);
	CHECK_ERROR("Launching trace kernel");

	cudaFree(d_raydirs);

	return 0;
}

__host__
int
cudamallocshapes(shape*& d_shapes,
				 shape* shapes,
				 int size)
{
	SAFE(cudaMalloc(&d_shapes, sizeof(shape)));
	SAFE(cudaMemcpy(d_shapes, shapes, size * sizeof(shape), cudaMemcpyHostToDevice));
	
	return 0;
}

__host__
int
cudamallocbuffer(unsigned char*& d_buffer,
				 int size)
{
	SAFE(cudaMalloc(&d_buffer, 3 * size * sizeof(unsigned char)));
	
	return 0;
}

__host__
int
cudareadbuffer(unsigned char* buffer,
               unsigned char* d_buffer,
			   int size)
{
	// Read results
	SAFE(cudaMemcpy(buffer, d_buffer, 3 * size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	return 0;
}