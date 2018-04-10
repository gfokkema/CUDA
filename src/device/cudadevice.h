#ifndef DEVICE_CUDADEVICE_H_
#define DEVICE_CUDADEVICE_H_

#include <curand.h>
#include <device/device.h>

class CudaDevice : public Device
{
public:
    CudaDevice(int pixels, int materials, int shapes);
    virtual ~CudaDevice();

    virtual void copy(std::vector<mat_t> materials, std::vector<shape_t> shapes);
    virtual double write(color_t * buffer, unsigned size);

    virtual double producerays(scene_t& scene, unsigned camsize, unsigned sample);
    virtual double pathtrace(scene_t& scene);
    virtual double rgbtoint(scene_t& scene, unsigned sample);
private:
    dims_t d_dimensions(camera_t& camera);
    virtual scene_t d_scene(Scene& scene);
private:
    curandGenerator_t d_generator;

    state_t d_state;
    output_t d_output;

    mat_t* d_mats;
    shape_t* d_shapes;

    int matsize;
    int shapesize;
};

#endif /* DEVICE_CUDADEVICE_H_ */
