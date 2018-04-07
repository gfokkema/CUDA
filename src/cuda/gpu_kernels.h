#ifndef __HOST_KERNELS_CUH
#define __HOST_KERNELS_CUH

#define CHECK_ERROR(errorMessage) {                                                 \
        cudaError_t err = cudaGetLastError();                                       \
        if( cudaSuccess != err) {                                                   \
            fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",       \
                    errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );   \
                    exit(EXIT_FAILURE);                                             \
        }                                                                           \
}

#ifdef __cplusplus
extern "C"
{
#endif

struct camera_t;
struct color_t;
struct float4;
struct ray_t;
struct scene_t;

int cudaproduceray(camera_t cam, ray_t* d_raydirs, float4* d_factor,
                   float4* d_result, float4* d_film, short samplecount);
int cudapathtrace(scene_t scene, ray_t* d_raydirs, float4* d_random,
                  float4* d_factor, float4* d_result);
int cudargbtoint(camera_t cam, float4* d_result, float4* d_film,
                 color_t* d_output, short samplecount);

#ifdef __cplusplus
}
#endif
#endif /** __HOST_KERNELS_CUH */
