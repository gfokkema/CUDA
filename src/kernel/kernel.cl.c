/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "kernel.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[2] = {
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 2, initBlocks, pair_map };

// Block function
void (^produceray_kernel)(const cl_ndrange *ndrange, camera* cam, cl_float4* output) =
^(const cl_ndrange *ndrange, camera* cam, cl_float4* output) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel produceray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, cam, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, output, &kargs);
  gcl_log_cl_fatal(err, "setting argument for produceray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing produceray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^traceray_kernel)(const cl_ndrange *ndrange, camera* cam, cl_float4* read_rays, shape* read_shapes, cl_uchar* write_buffer) =
^(const cl_ndrange *ndrange, camera* cam, cl_float4* read_rays, shape* read_shapes, cl_uchar* write_buffer) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel traceray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, cam, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, read_rays, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, read_shapes, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, write_buffer, &kargs);
  gcl_log_cl_fatal(err, "setting argument for traceray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing traceray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/kernel.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == produceray_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "produceray", &err);
          assert(bmap.map[1].block_ptr == traceray_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "traceray", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = produceray_kernel;
  bmap.map[1].block_ptr = traceray_kernel;
}

