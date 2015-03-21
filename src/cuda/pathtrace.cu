#include "gpu_kernels.h"

#define RANDOM 12345

__device__
bool
intersect(ray_t& ray, shape_t& shape, hit_t* hit)
{
    float4 trans_origin = ray.pos - shape.sphere.origin;
    float a = dot(ray.dir, ray.dir);
    float b = 2 * dot(trans_origin, ray.dir);
    float c = dot(trans_origin, trans_origin) - shape.sphere.radius * shape.sphere.radius;

    float disc = b * b - 4 * a * c;
    if (disc < 0)       return false;

    // We use the following in place of the quadratic formula for
    // more numeric precision.
    float q = (b > 0) ?
            -0.5 * (b + sqrt(disc)) :
            -0.5 * (b - sqrt(disc));
    float t0 = q / a;
    float t1 = c / q;
    //if (t0 < t1) swap(t0,t1);

    float t;
    if (t0 < EPSILON)   return false;
    if (t1 < 0)         t = t0;
    else                t = t1;

    hit->pos    = ray.pos + t * ray.dir;
    hit->normal = normalize(hit->pos - shape.sphere.origin);
    hit->matidx = shape.matidx;

    return true;
}

__global__
void
pathtraceray(camera_t         cam,
             ray_t*           d_raydirs,
             float4*          d_reflect,
             float4*          d_result,
             float4*          d_random,
             mat_t*           d_materials,
             shape_t*         d_shapes, int num_shapes)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx = (yi * cam.width + xi);

    if (d_raydirs[idx].dir.w < 0) return;

    float dist = FLT_MAX;
    hit_t hit;
    for (int i = 0; i < num_shapes; i++)
    {
        hit_t new_hit;
        if (intersect(d_raydirs[idx], d_shapes[i], &new_hit))
        {
            float new_dist = length(new_hit.pos - cam.pos);
            if (new_dist < dist)
            {
                dist = new_dist;
                hit  = new_hit;
            }
        }
    }

    // Check whether this ray intersected the scene, if not kill the ray
    if (dist >= FLT_MAX)
    {
        d_raydirs[idx].dir.w = -1;
        return;
    }

    // Should be: emission + color * (recursive path trace)
    mat_t* mat     = d_materials + hit.matidx;
    d_result[idx]  = d_result[idx] + d_reflect[idx] * mat->emit;   // ACCUMULATE COLOR USING REFLECT AND EMIT
    d_reflect[idx] =                 d_reflect[idx] * mat->color;

    // Reflect, refract or both
    if (mat->type == MIRROR)
    {
        d_raydirs[idx].dir = reflect(d_raydirs[idx].dir, hit.normal);
    }
    else if (mat->type == DIFFUSE)
    {
        unsigned randidx = (idx + (int)dot(d_raydirs[idx].dir, d_raydirs[idx].pos)) % (cam.width * cam.height);
        d_raydirs[idx].dir = randvector(d_random[randidx], hit.normal);
    }
    else if (mat->type == TRANSPARENT)
    {
        // calculate refraction and reflection
        printf("should not print!");
    }
    d_raydirs[idx].pos = hit.pos + EPSILON * d_raydirs[idx].dir;
}

int
cudapathtrace(camera_t        cam,
              ray_t*          d_raydirs,
              float4*         d_reflect,
              float4*         d_result,
              float4*         d_random,
              mat_t*          d_materials,
              shape_t*        d_shapes, int num_shapes)
{
    unsigned size = cam.height * cam.width;

    curandGenerator_t gen;
    SAFE_RAND( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    SAFE_RAND( curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) );
    SAFE_RAND( curandGenerateUniform(gen, (float*)d_random, 4 * size) );

    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    for (int i = 0; i < 5; i++)
    {
        pathtraceray <<< numblocks,threadsperblock >>> (cam,
                                                        d_raydirs,
                                                        d_reflect,
                                                        d_result,
                                                        d_random,
                                                        d_materials,
                                                        d_shapes, num_shapes);
    }
    CHECK_ERROR("Launching pathtrace kernel");

    return 0;
}

__global__
void
rgbtoint(camera_t      cam,
         float4*       d_result,
         color_t*      d_buffer)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx = (yi * cam.width + xi);

    d_buffer[idx].r = d_result[idx].x * 255;
    d_buffer[idx].g = d_result[idx].y * 255;
    d_buffer[idx].b = d_result[idx].z * 255;
}

int cudargbtoint(camera_t        cam,
                 float4*         d_result,
                 color_t*        d_buffer)
{
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    rgbtoint <<< numblocks,threadsperblock >>> (cam,
                                                d_result,
                                                d_buffer);
    CHECK_ERROR("Launching rgbtoint kernel");

    return 0;
}
