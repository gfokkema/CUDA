#ifndef DEVICE_H_
#define DEVICE_H_

#include <vector>

#include "util/camera.h"
#include "devices/device_mem.h"

enum kernel_key : uint8_t {
	KERNEL_PRODUCE_RAY,
	KERNEL_TRACE_RAY,

	/* Keep last */
	KERNEL_COUNT
};

class Device {
public:
	virtual ~Device() {};

	virtual int init() = 0;

	virtual device_mem malloc(size_t size, permission perm) = 0;
	virtual void read(device_mem mem, size_t size, void* data_read) = 0;
	virtual void write(device_mem mem, size_t size, void* data_write) = 0;
	virtual int enqueue_kernel_range(kernel_key id, uint8_t num_args, void** arg_values,
					size_t* arg_sizes, uint8_t dim, size_t* work_size) = 0;
};

#endif /* DEVICE_H_ */
