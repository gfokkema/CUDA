/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
typedef struct camera {
  cl_float4 pos;
  cl_float4 dir;
  cl_float4 up;
  cl_float4 right;
  cl_int width;
  cl_int height;
  cl_int _;
  cl_int __;
} _camera_unalign;

typedef _camera_unalign __attribute__ ((aligned(8))) camera;

typedef struct _sphere {
  cl_float4 origin;
  cl_float radius;
} __sphere_unalign;

typedef __sphere_unalign __attribute__ ((aligned(8))) _sphere;

typedef struct _plane {
  cl_float4 origin;
  cl_float4 normal;
} __plane_unalign;

typedef __plane_unalign __attribute__ ((aligned(8))) _plane;

typedef struct _triangle {
  cl_float4 v1;
  cl_float4 v2;
  cl_float4 v3;
} __triangle_unalign;

typedef __triangle_unalign __attribute__ ((aligned(8))) _triangle;

typedef union _shape_data {
  __sphere_unalign sp;
  __plane_unalign pl;
  __triangle_unalign tr;
} __shape_data_unalign;

typedef __shape_data_unalign __attribute__ ((aligned(8))) _shape_data;

typedef struct shape {
  __shape_data_unalign data;
  cl_int type;
} _shape_unalign;

typedef _shape_unalign __attribute__ ((aligned(8))) shape;

extern void (^produceray_kernel)(const cl_ndrange *ndrange, camera* cam, cl_float4* output);
extern void (^traceray_kernel)(const cl_ndrange *ndrange, camera* cam, cl_float4* read_rays, shape* read_shapes, cl_uchar* write_buffer);
