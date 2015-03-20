#ifndef __GPU_VECTOR_CUH
#define __GPU_VECTOR_CUH

__device__ inline
float4 operator+(const float4& lhs, const float4& rhs) {
    float4 retval;
    retval.x = lhs.x + rhs.x;
    retval.y = lhs.y + rhs.y;
    retval.z = lhs.z + rhs.z;
    return retval;
}

__device__ inline
float4 operator-(const float4& lhs, const float4& rhs) {
    float4 retval;
    retval.x = lhs.x - rhs.x;
    retval.y = lhs.y - rhs.y;
    retval.z = lhs.z - rhs.z;
    return retval;
}

__device__ inline
float4 operator*(const float& lhs, const float4& rhs) {
    float4 retval;
    retval.x = lhs * rhs.x;
    retval.y = lhs * rhs.y;
    retval.z = lhs * rhs.z;
    return retval;
}

__device__ inline
float4 operator*(const float4& lhs, const float& rhs) {
    return rhs * lhs;
}

__device__ inline
float4 operator/(const float4& lhs, const float& rhs) {
    float4 retval;
    retval.x = lhs.x / rhs;
    retval.y = lhs.y / rhs;
    retval.z = lhs.z / rhs;
    return retval;
}

__device__ inline
float operator*(const float4& lhs, const float4& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__ inline
float length(const float4& lhs) {
    return sqrt(lhs * lhs);
}

__device__ inline
float4 normalize(const float4& lhs) {
    return lhs / length(lhs);
}

#endif /** __GPU_VECTOR_CUH */
