#include <devices/cpudevice.h>

CPUDevice::CPUDevice() {}

CPUDevice::~CPUDevice() {}

int CPUDevice::init() {
	return 0;
}

int CPUDevice::produceray(Camera* cam, float4*& raydirs) {
	raydirs = new float4[cam->width() * cam->height()];

	float invwidth = 1.f / cam->width();
	float invheight = 1.f / cam->height();

	float4 *ray = raydirs;
	for (int yi = 0; yi < cam->height(); yi++) {
		for (int xi = 0; xi < cam->width(); xi++) {
			float x = (xi + .5) * invwidth - 0.5;
			float y = (yi + .5) * invheight - 0.5;

			*ray++ = (x * cam->right() + y * cam->up() + cam->dir()).normalize().gpu_type();
		}
	}
}

int CPUDevice::traceray(Camera *cam, float4* raydirs, std::vector<shape> shapes, unsigned char*& buffer) {
	for (int i = 0; i < cam->width() * cam->height(); i++) {
		for (shape shape : shapes) {
			buffer[i * 3] = intersect(cam->pos().gpu_type(), raydirs[i], shape);
		}
	}
	delete raydirs;
}

unsigned char CPUDevice::intersect(Vector origin, Vector dir, shape shape) {
	Vector trans_origin = origin - shape.sphere.origin;
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
	if (t0 < t1) std::swap(t0,t1);

	float t;
	if (t0 < EPSILON)	return 0;
	if (t1 < 0)		t = t0;
	else			t = t1;

	return 255;
}
