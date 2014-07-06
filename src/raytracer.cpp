#include <fstream>
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

	Vector *image = new Vector[WIDTH * HEIGHT], *pixel = image;
	for (int yi = 0; yi < HEIGHT; yi++) {
		for (int xi = 0; xi < WIDTH; xi++, pixel++) {
			float x = (xi + .5) * invwidth - 0.5;
			float y = (yi + .5) * invheight - 0.5;

			Vector imageproj = x * right + y * up + pos + dir;
			Vector raydir = (imageproj - pos).normalize();
			Ray ray(pos, raydir);

			*pixel = scene.trace(ray);
		}
	}

	std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
	for (unsigned i = 0; i < WIDTH * HEIGHT; i++) {
		ofs <<
		(unsigned char)(std::min(1.f, image[i][0]) * 255) <<
		(unsigned char)(std::min(1.f, image[i][1]) * 255) <<
		(unsigned char)(std::min(1.f, image[i][2]) * 255);
	}
	ofs.close();

	return 0;
}
