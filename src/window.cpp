#include <GL/glew.h>
#include <GL/gl.h>

#include <fstream>
#include <iostream>

#include "scene.h"

#define WIDTH 512
#define HEIGHT 512

Scene scene;

int main(int argc, char* argv[]) {
	//glewInit();
	//GLuint BUFFER;
	//glGenBuffers(1, &BUFFER);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, BUFFER);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT, NULL, GL_DYNAMIC_DRAW);

	Vector *image = new Vector[WIDTH * HEIGHT];
	scene.render(image, WIDTH, HEIGHT);

	//unsigned char* buffer_map = (unsigned char*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	//glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

	std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
	for (unsigned i = 0; i < WIDTH * HEIGHT; i++) {
		ofs <<
		(unsigned char)(std::min(1.f, image[i][0]) * 255) <<
		(unsigned char)(std::min(1.f, image[i][1]) * 255) <<
		(unsigned char)(std::min(1.f, image[i][2]) * 255);
	}
	ofs.close();
}
