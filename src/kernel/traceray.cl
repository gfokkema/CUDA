#ifndef __OPENCL_VERSION__
#include "opencl.h"
#endif

typedef struct cl_sphere {
	float radius;
	float4 origin;
} cl_shape;

__kernel void traceray(__global float4* read_rays, __global cl_shape* read_shapes, __global unsigned char* write_buffer, __const float4 origin) {
	int i = get_global_id(0);
	printf("debug %d\n", i);
}

__kernel void intersect(__local float4* ray, __local* shape, __const float4 origin) {

}
