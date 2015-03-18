#include "device_util.cuh"
#include "host_kernels.cuh"

__host__
int
cudamallocshapes(shape_t*& d_shapes,
                 shape_t* shapes,
                 int size)
{
    SAFE(cudaMalloc(&d_shapes, sizeof(shape_t)));
    SAFE(cudaMemcpy(d_shapes, shapes, size * sizeof(shape_t), cudaMemcpyHostToDevice));

    return 0;
}

__host__
int
cudamallocbuffer(unsigned char*& d_buffer,
                 int size)
{
    SAFE(cudaMalloc(&d_buffer, 3 * size * sizeof(unsigned char)));

    return 0;
}

__host__
int
cudareadbuffer(unsigned char* buffer,
               unsigned char* d_buffer,
               int size)
{
    // Read results
    SAFE(cudaMemcpy(buffer, d_buffer, 3 * size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    return 0;
}
