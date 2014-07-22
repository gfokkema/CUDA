#include "session.h"
#include <GL/glew.h>
#include <cstdint>
#include <ctime>

RenderSession::RenderSession(Device* device, Scene* scene) : _device(device), _scene(scene)
{
	size_t cam_size = _scene->_cam->size();

	// Allocate memory for the cam struct (eventhough it's just one struct).
	this->cam_buff = _device->malloc(sizeof(camera), PERM_READ_ONLY);

	// Allocate memory on the device for the ray directions.
	this->ray_dirs = _device->malloc(cam_size * sizeof(float4), PERM_READ_WRITE);

	// Allocate memory for the shape buffer.
	this->shapes = _device->malloc(_scene->_shapes.size() * sizeof(shape), PERM_READ_ONLY);
	// Perform a blocking write of the shape data to the buffer that was just allocated.
	_device->write(this->shapes, _scene->_shapes.size() * sizeof(shape), _scene->_shapes.data());

	// Allocate memory for the buffer that's to be written to.
	this->buffer = _device->malloc(3 * cam_size * sizeof(unsigned char), PERM_WRITE_ONLY);
}

void print_camera_info(camera cam) {
	std::cout<<"Cam attributes:\n"
		 <<"  pos:   "<< cam.pos.x<<','<<cam.pos.y<<','<<cam.pos.z<<','<<cam.pos.w<<"\n"
		 <<"  dir:   "<< cam.dir.x<<','<<cam.dir.y<<','<<cam.dir.z<<','<<cam.dir.w<<"\n"
		 <<"  right: "<< cam.right.x<<','<<cam.right.y<<','<<cam.right.z<<','<<cam.right.w<<"\n"
		 <<"  width, height: "<< cam.width << ',' << cam.height <<"\n"
		 <<"  size: "<< sizeof(camera)
		 <<std::endl;
}

void RenderSession::render() {
	//Time the rendering process
//	std::clock_t c_start = std::clock();

	glClear(GL_COLOR_BUFFER_BIT);

	camera cam = _scene->_cam->gpu_type();
	// Copy the camera data to the cam buffer.
	_device->write(this->cam_buff, sizeof(camera), (void *) &cam);

	size_t pr_work_size = _scene->_cam->height();
	size_t cam_size = _scene->_cam->size();

	// Arguments: cl_camera cam, float4* output
	void* pr_arg_values[2] = { (void *) &cam_buff._mem_pointer, (void *) &(ray_dirs._mem_pointer) };
	size_t pr_arg_sizes[2] = { cam_buff._mem_size, ray_dirs._mem_size };

	// Blocking call!
	_device->enqueue_kernel_range(KERNEL_PRODUCE_RAY, 2, pr_arg_values, pr_arg_sizes, 1, &pr_work_size);

	// Arguments: cl_camera cam, float4* read_rays, shape* read_shapes, unsigned char* write_buffer
	void* tr_arg_values[4] = { (void *) &cam_buff._mem_pointer, &ray_dirs._mem_pointer, &shapes._mem_pointer, &buffer._mem_pointer };
	size_t tr_arg_sizes[4] = { cam_buff._mem_size, ray_dirs._mem_size, shapes._mem_size, buffer._mem_size };

	//print_camera_info(cam);
	// Blocking call!
	_device->enqueue_kernel_range(KERNEL_TRACE_RAY, 4, tr_arg_values, tr_arg_sizes, 1, &cam_size);

	// TODO: remainder should probably be in a different function
	unsigned char* buffer_result = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	// Read the buffer.
	_device->read(this->buffer, 3 * cam_size * sizeof(unsigned char), buffer_result);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

//	std::clock_t c_end = std::clock();
//	std::cout << "Frame duration:\t" << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\r";
//	std::flush(std::cout);
}
