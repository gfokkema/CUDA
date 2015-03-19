#include <cuda_runtime.h>
#include <cfloat>

#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#define EPSILON 1e-4

enum type {
    SPHERE,
    PLANE,
    TRIANGLE
};

typedef struct color_t {
    unsigned char r, g, b;
} color_t;

typedef struct shape_t {
    union {
        // SPHERE
        struct {
            float4 origin;	// offset 0
            float radius;	// offset 16
        } sphere;
        // PLANE
        struct {
            float4 origin;	// offset 0
            float4 normal;	// offset 16
        } plane;
        // TRIANGLE
        struct {
            float4 v1;		// offset 0
            float4 v2;		// offset 16
            float4 v3;		// offset 32
        } triangle;
    };

    int     type;                // offset 48
    float   emit;
    color_t color;
} shape_t;

typedef struct camera_t
{
    int width, height;
    float4 pos;
    float4 dir;
    float4 up;
    float4 right;
} camera_t;

typedef struct ray_t
{
    float4 pos;
    float4 dir;
} ray_t;

typedef struct hit_t
{
    float4 pos;
    float4 normal;
    float4 color;
    shape_t object;
} hit_t;

#endif /* GPU_TYPES_H_ */
