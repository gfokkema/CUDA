#ifndef __HOST_KERNELS_CUH
#define __HOST_KERNELS_CUH

#ifdef __cplusplus
extern "C"
{
#endif

struct dims_t;
struct scene_t;
struct state_t;
struct output_t;
struct curandGenerator_st;

void cudaproduceray(dims_t dim, state_t state, scene_t scene);
void cudapathtrace(dims_t dim, state_t state, scene_t scene, curandGenerator_st* gen);
void cudargbtoint(dims_t dim, state_t state, scene_t scene, output_t output, short samplecount);

#ifdef __cplusplus
}
#endif
#endif /** __HOST_KERNELS_CUH */
