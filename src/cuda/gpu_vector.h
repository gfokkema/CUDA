#ifndef __GPU_VECTOR_CUH
#define __GPU_VECTOR_CUH

#include <crt/host_defines.h>
#include <vector_types.h>

__device__ inline
float4 operator+(const float4& lhs, const float4& rhs)
{
    float4 retval;
    retval.x = lhs.x + rhs.x;
    retval.y = lhs.y + rhs.y;
    retval.z = lhs.z + rhs.z;
    return retval;
}

__device__ inline
float4 operator-(const float4& lhs)
{
    float4 retval;
    retval.x = -lhs.x;
    retval.y = -lhs.y;
    retval.z = -lhs.z;
    return retval;
}

__device__ inline
float4 operator-(const float4& lhs, const float4& rhs)
{
    float4 retval;
    retval.x = lhs.x - rhs.x;
    retval.y = lhs.y - rhs.y;
    retval.z = lhs.z - rhs.z;
    return retval;
}

__device__ inline
float4 operator*(const float4& lhs, const float4& rhs)
{
    float4 retval;
    retval.x = lhs.x * rhs.x;
    retval.y = lhs.y * rhs.y;
    retval.z = lhs.z * rhs.z;
    return retval;
}

__device__ inline
float4 operator*(const float& lhs, const float4& rhs)
{
    float4 retval;
    retval.x = lhs * rhs.x;
    retval.y = lhs * rhs.y;
    retval.z = lhs * rhs.z;
    return retval;
}

__device__ inline
float4 operator*(const float4& lhs, const float& rhs)
{
    return rhs * lhs;
}

__device__ inline
float4 operator/(const float4& lhs, const float& rhs)
{
    float4 retval;
    retval.x = lhs.x / rhs;
    retval.y = lhs.y / rhs;
    retval.z = lhs.z / rhs;
    return retval;
}

__device__ inline
float dot(const float4& lhs, const float4& rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__ inline
float length(const float4& lhs)
{
    return sqrt(dot(lhs, lhs));
}

__device__ inline
float4 normalize(const float4& lhs)
{
    return lhs / length(lhs);
}

__device__ inline
float4 reflect(const float4& raydir, const float4& normal)
{
    return raydir - 2 * dot(raydir, normal) * normal;
}

__device__ inline
float4 randvector(const float4& randray, const float4& normal)
{
    float x = 2 * randray.x - 1;
    float r = sqrt(1 - x * x);
    float theta = randray.y * 2 * M_PI;

    float4 retval = { x, sinf(theta) * r, cosf(theta) * r, 0 };
    if (dot(retval, normal) < 0)
        retval = -retval;

    return retval;
}

#endif /** __GPU_VECTOR_CUH */
