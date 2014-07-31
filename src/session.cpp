#include <GL/glew.h>
#include "session.h"
#include <iomanip>
#include <cstdint>

RenderSession::RenderSession(Device* device, Scene* scene) : _device(device), _scene(scene)
{
	size_t cam_size = _scene->_cam->size();

	// Allocate memory on the device
	this->ray_dirs = _device->malloc(cam_size * sizeof(gpu_float4), MEM_TYPE_READ_WRITE);

	// Allocate memory for the shape buffer.
	this->shapes = _device->malloc(_scene->_shapes.size() * sizeof(shape), _scene->_shapes.data(), MEM_TYPE_READ_ONLY | MEM_TYPE_COPY_HOST_PTR);
	// Perform a blocking write of the shape data to the buffer that was just allocated.
	//_device->write(this->shapes, _scene->_shapes.size() * sizeof(shape), _scene->_shapes.data());

	// Allocate memory for the buffer that's to be written to.
	this->buffer = _device->malloc(3 * cam_size * sizeof(unsigned char), MEM_TYPE_WRITE_ONLY);

	_start = std::chrono::system_clock::now();
}

RenderSession::~RenderSession() {
	std::chrono::time_point<std::chrono::system_clock> _end;
	_end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = _end-_start;

	std::cout<<"Elapsed time: "<< elapsed_seconds.count() << "s\n";
	std::cout<<"Frames: "<<frames<<"\n";
	std::cout<<"Average FPS: "<<(frames/elapsed_seconds.count()) << std::endl;
}

void RenderSession::render() {
	//Time the rendering process
	std::chrono::time_point<std::chrono::system_clock> start;
	start = std::chrono::system_clock::now();

	glClear(GL_COLOR_BUFFER_BIT);

	camera cam = _scene->_cam->gpu_type();
	device_mem cam_buff = _device->malloc(sizeof(camera), &cam, MEM_TYPE_READ_ONLY | MEM_TYPE_COPY_HOST_PTR);
	size_t pr_work_size = _scene->_cam->height();
	size_t cam_size = _scene->_cam->size();
	int num_shapes = _scene->_shapes.size();

	// Arguments: cl_camera cam, float4* output
	void* pr_arg_values[2] = {
		&cam_buff._mem_pointer,
		&ray_dirs._mem_pointer
	};
	size_t pr_arg_sizes[2] = { cam_buff._mem_size, ray_dirs._mem_size };

	// Blocking call!
	_device->enqueue_kernel_range(KERNEL_PRODUCE_RAY, 2, pr_arg_values, pr_arg_sizes, 1, &pr_work_size);

	// Arguments: cl_camera cam, float4* read_rays, shape* read_shapes, unsigned char* write_buffer
	void* tr_arg_values[5] = {
		&cam_buff._mem_pointer,
		&ray_dirs._mem_pointer,
		&shapes._mem_pointer,
		&num_shapes,
		&buffer._mem_pointer
	};
	size_t tr_arg_sizes[5] = { cam_buff._mem_size, ray_dirs._mem_size, shapes._mem_size, sizeof(int), buffer._mem_size };

	// Blocking call!
	_device->enqueue_kernel_range(KERNEL_TRACE_RAY, 5, tr_arg_values, tr_arg_sizes, 1, &cam_size);

	// TODO: remainder should probably be in a different function
	unsigned char* buffer_result = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	// Read the buffer.
	_device->read(this->buffer, 3 * cam_size * sizeof(unsigned char), buffer_result);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

	std::chrono::time_point<std::chrono::system_clock> end;
	end = std::chrono::system_clock::now();
	std::chrono::duration<double, std::milli> delta_time = end - start;
	std::chrono::duration<double> delta_time_sec = end - start;

	std::cout << "\e[7mFrame duration:\t" << std::setw(5) << delta_time.count() << " ms"<< "\tFramerate:\t" << std::setw(5) << 1 / delta_time_sec.count() << " fps\r";
	std::flush(std::cout);

	// GLOBAL COUNTER
	frames++;
}
