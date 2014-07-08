#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sys/time.h>

#include "util/camera.h"
#include "scene.h"

#define WIDTH 640
#define HEIGHT 480

namespace {
	GLFWwindow* window;
	Camera camera(WIDTH, HEIGHT);
	Scene scene;

	/**
	* Time independent keyboard function
	*/
	void key_callback(GLFWwindow * window, int key, int scancode, int action, int mods) {
		switch (key) {
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, 1);
				break;
		}
	}

	void handle_mouse() {
		double x, y;
		double middle_x = WIDTH/2.0;
		double middle_y = HEIGHT/2.0;
		glfwGetCursorPos(window, &x, &y);
		if (x == 0.f || y == 0.f) return;
		double dx = x - middle_x;
		double dy = y - middle_y;
		if (dx == 0.f && dy == 0.f) return;
		std::cout<<x<<", "<<y<<std::endl;
		scene.cam.lookAt(x, HEIGHT - y);
		glfwSetCursorPos(window, middle_x, middle_y);
	}

	/**
	* Time dependent keyboard function
	*/
	void handle_keyboard(float dt) {
		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) camera.strafe(-1.f, dt);
		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) camera.strafe(1.f, dt);
		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) camera.move(1.f, dt);
		if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) camera.move(-1.f, dt);
	}

	void handle_input(float dt) {
		handle_keyboard(dt);
		handle_mouse();
	}
}

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

	scene.setCamera(&camera);

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set the keyboard callback for time independent keyboard handling
	glfwSetKeyCallback(window, &key_callback);
	double middle_x = WIDTH/2.0;
	double middle_y = HEIGHT/2.0;
	glfwSetCursorPos(window, middle_x, middle_y);

	// Initialize our vertex buffer
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 3, 0, GL_DYNAMIC_DRAW);

	do {
		// Set the timer to zero
		glfwSetTime(0.0);
		glClear(GL_COLOR_BUFFER_BIT);

		// Map the buffer and render the scene
		unsigned char* buffer = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
		scene.render(buffer);
		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

		// Draw the buffer onto the off screen buffer
		glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

		double dt = glfwGetTime();
		handle_input(dt);
//		printf("FPS: %f\n", 1.f / dt);
	} while(!glfwWindowShouldClose(window));

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}
