#ifndef __OPENCL_VERSION__
#include "opencl.h"
#endif

#define EPSILON 1e-4

typedef struct cl_camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
} cl_camera;

typedef struct cl_shape {
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
} cl_shape;

bool intersect(__const float4 origin, float4 dir, cl_shape shape);

__kernel void produceray(__global float4* output, __const cl_camera cam) {
	int yi = get_global_id(0);
	int offset = yi * cam.width;

	float invwidth = 1.f / cam.width;
	float invheight = 1.f / cam.height;

	for (int xi = 0; xi < cam.width; xi++) {
		float x = (xi + .5) * invwidth - 0.5;
		float y = (yi + .5) * invheight - 0.5;

		output[offset + xi] = normalize(x * cam.right + y * cam.up + cam.dir);
	}
}

__kernel void traceray(__global float4* read_rays, __global cl_shape* read_shapes, __global unsigned char* write_buffer, __const float4 origin) {
	int idx = get_global_id(0);

	if (intersect(origin, read_rays[idx], read_shapes[0]))
		write_buffer[idx * 3] = 255;
}

bool intersect(__const float4 origin, float4 dir, cl_shape shape) {
	float4 trans_origin = origin - shape.sphere.origin;
	float a = dot(dir, dir);
	float b = 2 * dot(trans_origin, dir);
	float c = dot(trans_origin, trans_origin) - dot(shape.sphere.radius, shape.sphere.radius);

	float disc = b * b - 4 * a * c;
	if (disc < 0)	return false;

	// We use the following in place of the quadratic formula for
	// more numeric precision.
	float q = (b > 0) ?
				-0.5 * (b + sqrt(disc)) :
				-0.5 * (b - sqrt(disc));
	float t0 = q / a;
	float t1 = c / q;

	float t;
	if (t0 < EPSILON)	return false;
	if (t1 < 0)		t = t0;
	else			t = t1;

	return true;
}
