#define __kernel
#define __global
#define __constant
#define __local

#define bool;
#define false;
#define true;

typedef struct float2 {
  float x, y;
} float2;
typedef struct float3 {
  float x, y, z;
  float2 xy, xz, yx, yz, zx, zy;
} float3;
typedef struct float4 {
  float x, y, z, w;
  float2 xy, yx;
  float3 xyz, xzy, yxz, yzx, zxy, zyx;
} float4;
