#ifndef __OPENCL_VERSION__
#include "gpu_types.h"
#ifndef __APPLE_KERNEL_COMPILE__
#include "opencl.h"
#endif /* __APPLE_KERNEL_COMPILE__ */
#endif /* __OPENCL_VERSION__ */

bool
plane_intersect(
			const ray ray,
			__constant plane *plane,
			float4 *new_origin,
			float4 *normal)
{
	*normal = plane->normal;
	*normal = normalize(*normal);
	float4 normal_deref = *normal;
	float4 plane_origin = plane->origin;

	float denom = dot(ray.dir,normal_deref);
	if (denom > -EPSILON && denom < EPSILON) return false;

	// Calculate term t in the expressen 'p = o + tD'
	float t = dot(plane_origin - ray.origin, normal_deref) / denom;
	if (t < EPSILON) return false;

	*new_origin = ray.origin + t * ray.dir;
	//return true;

	float checker_size = 0.5f;
	int u = (*new_origin)[0]/checker_size;
	int v = (*new_origin)[2]/checker_size;
	char uv_even = (u + v) % 2;
	char mask_uv = uv_even >> 7;
	unsigned char abs_uv_even = (uv_even ^ mask_uv) - mask_uv;
	bool even = (bool) abs_uv_even;
	return even;
}

bool
sphere_intersect(
			const ray ray,
			__constant sphere *sphere,
			float4 *new_origin,
			float4 *normal)
{
	float4 trans_origin = ray.origin - sphere->origin;
	float a = dot(ray.dir, ray.dir);
	float b = 2 * dot(trans_origin, ray.dir);
	float c = dot(trans_origin, trans_origin) - dot(sphere->radius, sphere->radius);

	float disc = b * b - 4 * a * c;
	if (disc < 0)	return false;

	// We use the following in place of the quadratic formula for
	// more numeric precision.
	float q = (b > 0) ?
				-0.5 * (b + sqrt(disc)) :
				-0.5 * (b - sqrt(disc));
	float t0 = q / a;
	float t1 = c / q;

	// FIXME: Does this work?
	// Swap the old value at address &t0 with t1, return the old value
	// at &t0.
	//if (t0 < t1) t1 = atomic_xchg(&t0, t1);
	if (t0 < t1) {
		float temp = t0;
		t0 = t1;
		t1 = temp;
	}

	float t;
	if (t0 < EPSILON)	return false;
	if (t1 < 0)		t = t0;
	else			t = t1;

	*normal = trans_origin + t * ray.dir;
	// FIXME:
	//*normal = normalize(*normal);
	*new_origin = ray.origin + t * ray.dir;

	return true;
}

bool
intersect(
		const ray ray,
		__constant shape *shape,
		float4 *new_origin,
		float4 *normal)
{
	switch (shape->type) {
		case SPHERE:
		return sphere_intersect(ray, &shape->data.sp, new_origin, normal);
		break;
		case PLANE:
		return plane_intersect(ray, &shape->data.pl, new_origin, normal);
		break;
		case TRIANGLE:
		//return triangle_intersect(origin, dir, &shape->data.tr, new_origin, normal);
		break;
	}
}

float4
shade(
		__constant shape *shape,
		const float4 cam_pos,
		float4 *intersect,
		float4 *light_pos,
		float4 *normal)
{
		float4 ambient = (float4)(0.f,0.f,0.f,0.f);
		float4 diffuse = (float4)(0.f,0.f,0.f,0.f);
		float4 specular = (float4)(0.f,0.f,0.f,0.f);
		float4 Kd = (float4)(1.f,0.f,0.f,0.f);
		float4 Ks = (float4)(1.f,1.f,1.f,1.f);

		/* Diffuse */
		float4 light_vec = *light_pos - *intersect;
		float4 normal_deref = *normal;
		normal_deref = normalize(normal_deref);
		light_vec = normalize(light_vec);
		float dot_prod = dot(normal_deref, light_vec);
		diffuse = dot_prod * Kd;

		/* Specular */
		float4 reflect = 2 * dot(light_vec, normal_deref) * normal_deref - light_vec;
		reflect = normalize(reflect);
		float4 view = cam_pos - *intersect;
		view = normalize(view);
		float dot_prod_spec = dot(view, reflect);
		if (dot_prod_spec >= 0) {
			float shininess = 10;
			specular = pow(dot_prod_spec, shininess) * Ks;
		}
		return ambient + diffuse + specular;
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

void fill_buffer(float4 color, __global unsigned char *write_buffer) {
	float4 final_color = clamp(color, 0.f, 1.f);	
	write_buffer[0] = (unsigned char) (final_color.s0 * 255);
	write_buffer[1] = (unsigned char) (final_color.s1 * 255);
	write_buffer[2] = (unsigned char) (final_color.s2 * 255);
}

__kernel void
traceray(
		__constant camera* cam,
		__global float4* read_ray_dirs,
		__constant shape* read_shapes,
		int num_shapes,
		__global unsigned char* write_buffer)
{
	int idx = get_global_id(0);

	float4 light_pos = (float4)(2.f,3.f,1.f,0.f);

	float current_depth = FLT_MAX;
	bool intersection = false;
	float4 new_origin;
	float4 normal;
	int shape_index;

	// TODO: add for-loop which loops though all the shapes (needs num_shapes argument)
	ray ray = { cam->pos, read_ray_dirs[idx] };
	for (int i = 0; i < num_shapes; i++) {
		float4 new_new_origin;
		float4 new_normal;
		if (intersect(ray, read_shapes + i, &new_new_origin, &new_normal)) {
			float new_depth = length(new_new_origin - cam->pos);
			if (new_depth < current_depth) {
				intersection = true;
				current_depth = new_depth;
				normal = new_normal;
				new_origin = new_new_origin;
				// Store shape index
				shape_index = i;
			}
		}
	}

	if (!intersection) {
		fill_buffer((float4)(0.f,0.f,0.f,0.f), (write_buffer + idx * 3));
		return;
	}

	// TODO:Calculate reflected ray
	fill_buffer(shade(read_shapes + shape_index, cam->pos, &new_origin, &light_pos, &normal), (write_buffer + idx * 3));
}
