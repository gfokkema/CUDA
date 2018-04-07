#include <common.h>
#include <device/device.h>

#ifndef DEVICE_CUDADEVICE_H_
#define DEVICE_CUDADEVICE_H_

class CudaDevice : public Device
{
public:
    CudaDevice(int pixels, int materials, int shapes);
    virtual ~CudaDevice();

    virtual void copy(std::vector<mat_t> materials, std::vector<shape_t> shapes);

    virtual double producerays(camera_t & camera, unsigned camsize, unsigned sample);
    virtual double pathtrace(camera_t & camera);
    virtual double rgbtoint(camera_t & camera, unsigned sample);
    virtual double write(color_t * buffer, unsigned size);
private:
    curandGenerator_t d_generator;

    ray_t* d_raydirs;
    float4* d_random;
    float4* d_factor;
    float4* d_result;
    float4* d_film;
    color_t* d_output;    // output color

    mat_t* d_mats;
    shape_t* d_shapes;

    int matsize;
    int shapesize;
};

#endif /* DEVICE_CUDADEVICE_H_ */
