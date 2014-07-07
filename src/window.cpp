#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sys/time.h>

#include "scene.h"

#define WIDTH 512
#define HEIGHT 512

GLFWwindow* window;

Scene scene;

int main(int argc, char* argv[]) {
	// Initialise GLFW
	if (!glfwInit()) {
		fprintf( stderr, "Failed to initialize GLFW\n" );
		return -1;
	}

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(WIDTH, HEIGHT, "Raytracer", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window.\n" );
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Initialize our vertex buffer
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 3, 0, GL_DYNAMIC_DRAW);

	// Our ray tracing loop
	struct timeval start, end;
	gettimeofday(&start, nullptr);
	do {
		glClear(GL_COLOR_BUFFER_BIT);

		// Map the buffer and render the scene
		unsigned char* buffer = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
		scene.render(buffer, WIDTH, HEIGHT);
		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

		// Draw the buffer onto the off screen buffer
		glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

		gettimeofday(&end, nullptr);
		float diff =	(end.tv_sec + end.tv_usec * 1e-6) -
						(start.tv_sec + start.tv_usec * 1e-6);
		printf("FPS: %f\n", 1.f / diff);
		std::swap(start, end);
	} while(glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
			glfwWindowShouldClose(window) == 0 );

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}
