#include "session.h"

Rendersession::RenderSession(Device* device, Scene* scene) : _device(device), _scene(scene)
{}

void RenderSession::render() {
	camera cam = _scene->_cam.gpu_type();
	size_t size = _scene->_cam->size();
	// Allocate memory on the device
	device_mem ray_dirs = _device->malloc(size * sizeof(cl_float4), PERM_READ_WRITE);

	// Arguments: float4* output, cl_camera cam
	void* pr_arg_values[2] = { &ray_dirs, &cam };
	size_t pr_arg_sizes[2] = { ray_dirs._mem_size, sizeof(camera) };

	// Blocking call!
	_device->enqueue_kernel_range(KERNEL_PRODUCE_RAY, 2, pr_arg_values, pr_arg_sizes, 1, &size);

	// Allocate memory for the shape buffer.
	device_mem shapes = _device->malloc(_scene->shapes.size() * sizeof(shape), PERM_READ_ONLY);
	// Perform a blocking write of the shape data to the buffer that was just allocated.
	_device->write(shapes, _scene->shapes.size() * sizeof(shape), shapes.data());

	// Allocate memory for the buffer that's to be written to.
	device_mem buffer = _device->malloc(3 * size * sizeof(unsigned char), PERM_WRITE_ONLY);

	// Arguments: float4 origin, float4* read_rays, shape* read_shapes, unsigned char* write_buffer
	void* tr_arg_values[4] = { &cam.pos, &ray_dirs, &shapes, &buffer };
	size_t tr_arg_sizes[4] = { sizeof(float4), ray_dirs._mem_size, shapes._mem_size, buffer._mem_size };

	// Blocking call!
	_device->enqueue_kernel_range(KERNEL_TRACE_RAY, 4, tr_arg_values, tr_arg_sizes, 1, &size);

	// TODO: remainder should probably be in a different function
	unsigned char* buffer_result = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	// Read the buffer.
	_device->read(buffer, 3 * size * sizeof(unsigned char), buffer_result);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	// Draw the buffer onto the off screen buffer
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);

	// Swap buffers
	glfwSwapBuffers(window);
}
