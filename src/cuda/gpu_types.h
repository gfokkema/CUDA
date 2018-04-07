#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#include <vector_types.h>

#define EPSILON 1e-6

enum mat_type
{
    DIFFUSE, TRANSPARENT, MIRROR
};

enum shape_type
{
    SPHERE, PLANE, TRIANGLE
};

typedef struct color_t
{
    unsigned char r, g, b;
} color_t;

typedef struct mat_t
{
    float4 color;        // rgb color
    float4 emit;         // rgb emittance color
    mat_type type;
} mat_t;

typedef struct shape_t
{
    union
    {
        // SPHERE
        struct
        {
            float4 origin; // offset 0
            float radius;  // offset 16
        } sphere;
        // PLANE
        struct
        {
            float4 origin; // offset 0
            float4 normal; // offset 16
        } plane;
        // TRIANGLE
        struct
        {
            float4 v1; // offset 0
            float4 v2; // offset 16
            float4 v3; // offset 32
        } triangle;
    };
    short matidx;
    shape_type type;
} shape_t;

typedef struct camera_t
{
    int width, height;
    float ratio;
    float4 pos;
    float4 dir;
    float4 up;
    float4 right;
} camera_t;

typedef struct scene_t
{
    int num_shapes;
    int num_materials;
    camera_t camera;
    shape_t* shapes;
    mat_t* materials;
} scene_t;

typedef struct ray_t
{
    float4 pos;
    float4 dir;
} ray_t;

typedef struct hit_t
{
    float4 pos;
    float4 normal;
    short matidx;
} hit_t;

#endif /* GPU_TYPES_H_ */
