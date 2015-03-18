#include "../util/gpu_types.h"

#ifndef __HOST_KERNELS_CUH
#define __HOST_KERNELS_CUH

extern "C" int cudamallocshapes(shape_t*&       d_shapes,
        shape_t*        shapes,
        int             size);
extern "C" int cudamallocbuffer(unsigned char*& d_buffer,
        int             size);
extern "C" int cudareadbuffer  (unsigned char*  buffer,
        unsigned char*  d_buffer,
        int size);
extern "C" int cudaproduceray  (camera_t       cam,
                                float4*&       d_raydirs);
extern "C" int cudatraceray    (camera_t        cam,
                                float4*         d_raydirs,
                                shape_t*        d_shapes,
                                unsigned char*  d_buffer);

#endif /** __HOST_KERNELS_CUH */
