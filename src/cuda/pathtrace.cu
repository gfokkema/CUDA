#include <cfloat>
#include <cstdio>
#include <cuda/bounce.h>
#include <cuda/gpu_kernels.h>
#include <cuda/gpu_types.h>
#include <cuda/gpu_vector.h>

__device__
bool
intersect(shape_t& shape, ray_t& ray, float4& loc)
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

    loc = ray.pos + t * ray.dir;

    return true;
}

__device__
bool
intersectscene(scene_t& scene, ray_t& ray, hit_t& hit)
{
    float4 loc;
    float dist = FLT_MAX;
    for (int i = 0; i < scene.num_shapes; i++)
    {
        if (intersect(scene.shapes[i], ray, loc))
        {
            float new_dist = length(loc - ray.pos);
            if (new_dist < dist)
            {
                dist       = new_dist;
                hit.pos    = loc;
                hit.normal = normalize(hit.pos - scene.shapes[i].sphere.origin);
                hit.mat    = scene.materials + scene.shapes[i].matidx;
            }
        }
    }

    return dist < FLT_MAX;
}

__global__
void
pathtraceray(scene_t scene, state_t state)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned idx   = state.idx = (yi * scene.camera.width + xi);
    float4& factor = state.d_factor[idx];
    if (factor.w < 0) return;
    ray_t& ray     = state.d_raydirs[idx];
    float4& result = state.d_result[idx];

    // Check whether this ray intersected the scene, if not kill the ray
    hit_t hit;
    if (!intersectscene(scene, ray, hit))
    {
        factor.w = -1;
        return;
    }

    // Did we hit a light source?
    if (length(hit.mat->emit) > EPSILON)
    {
        result = result + factor * hit.mat->emit;
        return;
    }

    // Reflect, refract or both
    bounce(state, ray, hit);
}

__host__
int
cudapathtrace(scene_t scene, state_t state)
{
    // Perform computation on device
    dim3 threadsperblock(8, 8);
    dim3 numblocks(scene.camera.width  / threadsperblock.x,
                   scene.camera.height / threadsperblock.y);
    for (int i = 0; i < 10; i++)
    {
        pathtraceray <<< numblocks,threadsperblock >>> (
            scene, state
        );
    }
    CHECK_ERROR("Launching pathtrace kernel");

    return 0;
}
