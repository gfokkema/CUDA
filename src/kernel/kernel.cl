#include <gpu_types.h>

unsigned char
intersect(
		__const float4 origin,
		float4 dir,
		shape shape)
{
	float4 trans_origin = origin - shape.sphere.origin;
	float a = dot(dir, dir);
	float b = 2 * dot(trans_origin, dir);
	float c = dot(trans_origin, trans_origin) - dot(shape.sphere.radius, shape.sphere.radius);

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

__kernel void
produceray(
		__constant struct camera* cam,
		__global float4* output)
{
	int yi = get_global_id(0);
	int offset = yi * cam->width;

	float invwidth = 1.f / cam->width;
	float invheight = 1.f / cam->height;

	float4 cam_up = cross(cam->right, cam->dir) * cam->height / ((float) cam->width);
	for (int xi = 0; xi < cam->width; xi++) {
		float x = (xi + .5) * invwidth - 0.5;
		float y = (yi + .5) * invheight - 0.5;

		output[offset + xi] = normalize(x * cam->right + y * cam_up + cam->dir);
	}
}

void
print_camera_info(__constant struct camera *cam)
{
	printf("Cam attributes (kernel):\n  pos:   %f,%f,%f,%f\n  dir:   %f,%f,%f,%f\n  right: %f,%f,%f,%f\n  width, height: %d,%d\n  size: %d\n", cam->pos.x,cam->pos.y,cam->pos.z,cam->pos.w, cam->dir.x,cam->dir.y,cam->dir.z,cam->dir.w,cam->right.x,cam->right.y,cam->right.z,cam->right.w,cam->width, cam->height, sizeof(camera));
}

__kernel void
traceray(
		__constant struct camera* cam,
		__global float4* read_rays,
		__global shape* read_shapes,
		__global unsigned char* write_buffer)
{
	int idx = get_global_id(0);
	//print_camera_info(cam);
	write_buffer[idx * 3] = intersect(cam->pos, read_rays[idx], read_shapes[0]);
	write_buffer[idx * 3 + 1] = 0;
	write_buffer[idx * 3 + 2] = 0;
}
