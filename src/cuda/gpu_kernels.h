#include <cfloat>
#include <cmath>
#include <curand.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "gpu_types.h"
#include "gpu_vector.h"

#ifndef __HOST_KERNELS_CUH
#define __HOST_KERNELS_CUH

#define SAFE( call) {                                                       \
        cudaError err = call;                                               \
        if( cudaSuccess != err) {                                           \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString( err) );         \
                    exit(EXIT_FAILURE);                                     \
        }                                                                   \
}
#define SAFE_RAND( call) {                                                  \
        curandStatus_t err = call;                                          \
        if( CURAND_STATUS_SUCCESS != err) {                                 \
            fprintf(stderr, "Curand error in file '%s' in line %i : %d.\n", \
                    __FILE__, __LINE__, err );                              \
                    exit(EXIT_FAILURE);                                     \
        }                                                                   \
}
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

int cudaproduceray(camera_t cam, ray_t* d_raydirs, float4* d_factor,
                   float4* d_result, float4* d_film, short samplecount);
int cudapathtrace(scene_t scene, ray_t* d_raydirs, float4* d_factor,
                  float4* d_result, float4* d_random);
int cudargbtoint(camera_t cam, float4* d_result, float4* d_film,
                 short samplecount, color_t* d_output);

#ifdef __cplusplus
}
#endif
#endif /** __HOST_KERNELS_CUH */
