#ifndef BOUNCE_H_
#define BOUNCE_H_

#include <cstdio>
#include <cuda/gpu_types.h>
#include <cuda/gpu_vector.h>

__device__
void
diffuse(state_t& state, ray_t& ray, hit_t& hit)
{
    // LAMBERT'S LAW:
    // Le  = rho / pi * dot(L, N)
    // rho = diffuse color 0..1
    // L   = incoming light direction
    // N   = surface normal
    // sample probability is area of hemisphere, 1/2pi
    // factor  = 1 / prob * Le
    //         = 2pi * rho / pi * dot(L, N)
    //         = 2 * rho * dot(L, N)
    state.d_factor[state.idx] = state.d_factor[state.idx] * 2 * hit.mat->color * dot(-ray.dir, hit.normal);
    ray.dir = randvector(state.d_random[state.idx], hit.normal);
    ray.pos = hit.pos + EPSILON * ray.dir;
}

__device__
void
transparent(state_t& state, ray_t& ray, hit_t& hit)
{
    // For transparent material, we need fresnel to calculate reflection, refraction etc.
    //ray.dir = ???
    ray.pos = hit.pos + EPSILON * state.d_raydirs[state.idx].dir;
    printf("should not print!");
}

__device__
void
mirror(state_t& state, ray_t& ray, hit_t& hit)
{
    ray.dir = reflect(ray.dir, hit.normal);
    ray.pos = hit.pos + EPSILON * ray.dir;
}

__device__
void
bounce(state_t& state, ray_t& ray, hit_t& hit)
{
    // Reflect, refract or both
    if (hit.mat->type == DIFFUSE)
    {
        diffuse(state, ray, hit);
    }
    else if (hit.mat->type == TRANSPARENT)
    {
        transparent(state, ray, hit);
    }
    else if (hit.mat->type == MIRROR)
    {
        mirror(state, ray, hit);
    }
}

#endif /* BOUNCE_H_ */
