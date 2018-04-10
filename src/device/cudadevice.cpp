#include <device/cudadevice.h>
#include <scene.h>

#define CHECK_ERROR(errorMessage) {                                                 \
        cudaError_t err = cudaGetLastError();                                       \
        if( cudaSuccess != err) {                                                   \
            fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",       \
                    errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );   \
                    exit(EXIT_FAILURE);                                             \
        }                                                                           \
}
#define SAFE( call) {                                                       \
        cudaError err = call;                                               \
        if( cudaSuccess != err) {                                           \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString( err) );         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
}
#define SAFE_RAND( call) {                                                  \
        curandStatus_t err = call;                                          \
        if (err != CURAND_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "Curand error in file '%s' in line %i : %d.\n", \
                    __FILE__, __LINE__, err );                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
}

CudaDevice::CudaDevice(int pixels, int matsize, int shapesize)
: d_mats(nullptr),
  d_shapes(nullptr),
  matsize(matsize),
  shapesize(shapesize)
{
    // Initialize CUDA with a NOOP call
    if (cudaFree(0) != cudaSuccess)
    {
        throw std::runtime_error("Failed to initialize CUDA.");
    }

    // RANDOM GENERATOR
    SAFE_RAND(curandCreateGenerator(&d_generator, CURAND_RNG_PSEUDO_DEFAULT));
    SAFE_RAND(curandSetPseudoRandomGeneratorSeed(d_generator, time(NULL)));

    // PER PIXEL BUFFERS
    SAFE(cudaMalloc(&d_state.rays, pixels * sizeof(ray_t)));
    SAFE(cudaMalloc(&d_state.random, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_state.factor, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_state.result, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_output.film, pixels * sizeof(float4)));
    SAFE(cudaMalloc(&d_output.output, pixels * sizeof(color_t)));

    // SCENE DESCRIPTION
    SAFE(cudaMalloc(&d_mats, matsize * sizeof(mat_t)));
    SAFE(cudaMalloc(&d_shapes, shapesize * sizeof(shape_t)));
}

CudaDevice::~CudaDevice()
{
    SAFE_RAND(curandDestroyGenerator(d_generator));
    SAFE(cudaFree(d_state.rays));
    SAFE(cudaFree(d_state.random));
    SAFE(cudaFree(d_state.factor));
    SAFE(cudaFree(d_state.result));
    SAFE(cudaFree(d_output.film));
    SAFE(cudaFree(d_output.output));
    SAFE(cudaFree(d_mats));
    SAFE(cudaFree(d_shapes));
}

void
CudaDevice::copy(std::vector<mat_t> materials, std::vector<shape_t> shapes)
{
    SAFE(cudaMemcpy(d_mats, materials.data(), materials.size() * sizeof(mat_t), cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_shapes, shapes.data(), shapes.size() * sizeof(shape_t), cudaMemcpyHostToDevice));
}

double
CudaDevice::write(color_t * buffer, unsigned size)
{
    SAFE(cudaMemcpy(buffer, d_output.output, size * sizeof(color_t), cudaMemcpyDeviceToHost));

    return 0.f;
}

double
CudaDevice::producerays(scene_t& scene, unsigned camsize, unsigned sample)
{
    cudaproduceray(d_dimensions(scene.camera), d_state, scene);
    CHECK_ERROR("Launching produce kernel.");

    return 0.f;
}

double
CudaDevice::pathtrace(scene_t& scene)
{
    cudapathtrace(d_dimensions(scene.camera), d_state, scene, d_generator);
    CHECK_ERROR("Launching pathtrace kernel.");

    return 0.f;
}

double
CudaDevice::rgbtoint(scene_t& scene, unsigned sample)
{
    cudargbtoint(d_dimensions(scene.camera), d_state, scene, d_output, sample);
    CHECK_ERROR("Launching rgbtoint kernel.");

    return 0.f;
}

dims_t
CudaDevice::d_dimensions(camera_t& camera)
{
    dim3 threads(8, 8);
    dim3 numblocks(camera.width / threads.x, camera.height / threads.y);
    return { threads, numblocks };
}

scene_t
CudaDevice::d_scene(Scene& scene)
{
    return { 8, 0, scene.camera().gpu_type(), d_shapes, d_mats }; // FIXME: hardcoded shape size
}
