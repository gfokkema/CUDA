#ifndef __OPENCL_VERSION__
#include "gpu_types.h"
#ifndef __APPLE_KERNEL_COMPILE__
#include "opencl.h"
#endif /* __APPLE_KERNEL_COMPILE__ */
#endif /* __OPENCL_VERSION__ */

unsigned char
plane_intersect(
			__const float4 origin,
			float4 dir,
			__constant shape *shape)
{
	float4 normal = shape->data.pl.normal;
	float4 plane_origin = shape->data.pl.origin;
	//normal.normalize();

	float denom = dot(dir,normal);
	if (denom > -EPSILON && denom < EPSILON) return 0;

	// Calculate term t in the expressen 'p = o + tD'
	float t = dot(plane_origin - origin, normal) / denom;
	if (t < EPSILON) return 0;
	float4 intersect = origin + t * dir;

	float checker_size = 0.5f;
	int u = intersect[0]/checker_size;
	int v = intersect[2]/checker_size;
	char uv_even = (u + v) % 2;
	char mask_uv = uv_even >> 7;
	unsigned char abs_uv_even = (uv_even ^ mask_uv) - mask_uv;

	return abs_uv_even * 255;
}

unsigned char
sphere_intersect(
			__const float4 origin,
			float4 dir,
			__constant shape *shape)
{
	float4 trans_origin = origin - shape->data.sp.origin;
	float4 radius = shape->data.sp.radius;
	float a = dot(dir, dir);
	float b = 2 * dot(trans_origin, dir);
	float c = dot(trans_origin, trans_origin) - dot(radius, radius);

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

unsigned char
intersect(
		__const float4 origin,
		float4 dir,
		__constant shape *shape)
{
	switch (shape->type) {
		case SPHERE:
		return sphere_intersect(origin, dir, shape);
		break;
		case PLANE:
		return plane_intersect(origin, dir, shape);
		break;
		case TRIANGLE:
		//return triangle_intersect(origin, dir, shape);
		break;
	}
}

__kernel void
produceray(
		__constant camera* cam,
		__global float4* output)
{
	int yi = get_global_id(0);
	int offset = yi * cam->width;

	float invwidth = 1.f / cam->width;
	float invheight = 1.f / cam->height;

	for (int xi = 0; xi < cam->width; xi++) {
		float x = (xi + .5) * invwidth - 0.5;
		float y = (yi + .5) * invheight - 0.5;

		output[offset + xi] = normalize(x * cam->right + y * cam->up + cam->dir);
	}
}

__kernel void
traceray(
		__constant camera* cam,
		__global float4* read_rays,
		__constant shape* read_shapes,
		__global unsigned char* write_buffer)
{
	int idx = get_global_id(0);
	write_buffer[idx * 3] = intersect(cam->pos, read_rays[idx], read_shapes);
	write_buffer[idx * 3 + 1] = 0;
	write_buffer[idx * 3 + 2] = 0;
}
