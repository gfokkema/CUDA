#ifndef CUDADEVICE_H_
#define CUDADEVICE_H_

#include <cuda_runtime_api.h>

#include "devices/device.h"

extern int cudaproduceray(dim3 blocks, dim3 threads, __const__ camera cam, float4* raydirs);
extern int cudatraceray(dim3 blocks, dim3 threads, __const__ camera cam, float4* read_rays, shape* read_shapes, unsigned char* write_buffer);

class CUDADevice : public Device {
public:
	CUDADevice();
	virtual ~CUDADevice();

	virtual int init();

	virtual device_mem malloc(size_t size, permission perm);
	virtual void read(device_mem mem, size_t size, void* data_read);
	virtual void write(device_mem mem, size_t size, void* data_write);
	virtual int enqueue_kernel_range(kernel_key id, uint8_t num_args, void** arg_values,
					size_t* arg_sizes, uint8_t dim, size_t* work_size);
};

#endif /* CUDADEVICE_H_ */
