#ifndef __COMMON_H
#define __COMMON_H

#define EPSILON 1e-4

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

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "util/camera.h"
#include "util/gpu_types.h"
#include "util/ray.h"
#include "util/vector.h"

#endif /** __COMMON_H */
