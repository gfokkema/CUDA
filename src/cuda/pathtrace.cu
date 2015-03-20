#include "gpu_kernels.h"

#define RANDOM 12345

__device__
bool
intersect(ray_t& ray, shape_t& shape, hit_t* hit)
{
    float4 trans_origin = ray.pos - shape.sphere.origin;
    float a = ray.dir * ray.dir;
    float b = 2 * trans_origin * ray.dir;
    float c = trans_origin * trans_origin - shape.sphere.radius * shape.sphere.radius;

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
    hit->normal = normalize(hit->pos);
    hit->object = shape;

    return true;
}

__global__
void
pathtraceray(camera_t         cam,
             color_t*         d_buffer,
             float4*          d_random,
             ray_t*           d_raydirs,
             shape_t*         d_shapes, int num_shapes)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx = (yi * cam.width + xi);

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

    if (dist >= FLT_MAX)
    {
        d_buffer[idx]     = (color_t){ 0, 0, 0 };
        return;
    }

    if (hit.object.emit > 0)
    {
        d_buffer[idx]     = hit.object.color;
    }
}

int
cudapathtrace(camera_t        cam,
              color_t*        d_buffer,
              float4*         d_random,
              ray_t*          d_raydirs,
              shape_t*        d_shapes, int num_shapes)
{
    unsigned size = cam.height * cam.width;

    curandGenerator_t gen;
    SAFE_RAND( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    SAFE_RAND( curandSetPseudoRandomGeneratorSeed(gen, RANDOM) );
    SAFE_RAND( curandGenerateUniform(gen, (float*)d_random, 4 * size) );

    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(cam.width / threadsperblock.x,
                   cam.height / threadsperblock.y);
    pathtraceray <<< numblocks,threadsperblock >>> (cam,
                                                    d_buffer,
                                                    d_random,
                                                    d_raydirs,
                                                    d_shapes, num_shapes);

    CHECK_ERROR("Launching pathtrace kernel");

    return 0;
}
