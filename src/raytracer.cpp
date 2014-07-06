#include <iostream>

#include "util/camera.h"
#include "util/ray.h"

#include "scene.h"

#define WIDTH 512
#define HEIGHT 512

Camera cam;
Scene scene;

int main() {
	Camera cam;
	Vector up = cam.up();
	Vector right = cam.right();
	Vector pos = cam.pos();
	Vector dir = cam.dir();

	float invwidth = 1.f / WIDTH;
	float invheight = 1.f / HEIGHT;

	for (int xi = 0; xi < WIDTH; xi++) {
		for (int yi = 0; yi < HEIGHT; yi++) {
			float x = (xi + .5) * invwidth - 0.5;
			float y = (yi + .5) * invheight - 0.5;

			Vector imageproj = x * right + y * up + pos + dir;
			Vector raydir = (imageproj - pos).normalize();
			Ray ray(pos, raydir);

			Vector color = scene.trace(ray);
		}
	}
	return 0;
}
