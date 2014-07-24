#ifndef __VECTOR_H_
#define __VECTOR_H_

#include <cmath>
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#include "../kernel/shape_type.h"
#include "../kernel/kernel.cl.h"
#define gpu_float4 cl_float4
#else
#include "gpu_types.h"
#endif /* __APPLE__ */

class Vector {
public:
	Vector() : v{0, 0, 0} {};
	Vector(float x, float y, float z) : v{x, y, z} {};
	Vector(gpu_float4 cl) : v{cl.v4[0], cl.v4[1], cl.v4[2]} {};
	~Vector() {};

	/**
	 * Basic vector operations
	 */
	const float operator[](int index) const {
		return v[index];
	};
	Vector operator+(const Vector& rhs) const {
		return Vector(v[0] + rhs[0], v[1] + rhs[1], v[2] + rhs[2]);
	}
	Vector operator-(const Vector& rhs) const {
		return Vector(v[0] - rhs[0], v[1] - rhs[1], v[2] - rhs[2]);
	}

	Vector& operator+=(const Vector& rhs) {
		v[0] += rhs[0];
		v[1] += rhs[1];
		v[2] += rhs[2];
		return (*this);
	}

	/**
	 * Extended vector operations
	 */
	const float length() const {
		return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	};
	const Vector normalize() const {
		float invlength = 1 / this->length();
		return Vector(v[0] * invlength, v[1] * invlength, v[2] * invlength);
	}
	const float operator*(const Vector& rhs) const {
		return	v[0] * rhs[0] +
				v[1] * rhs[1] +
				v[2] * rhs[2];
	}
	const Vector operator%(const Vector& rhs) const {
		return Vector(	v[1] * rhs[2] - v[2] * rhs[1],
						v[2] * rhs[0] - v[0] * rhs[2],
						v[0] * rhs[1] - v[1] * rhs[0]);
	}
	const gpu_float4 gpu_type() const { return {v[0], v[1], v[2], 0}; };
private:
	float v[3];
};

inline Vector operator*(const float factor, const Vector& rhs) {
	return Vector(factor * rhs[0], factor * rhs[1], factor * rhs[2]);
}
inline Vector operator*(const Vector& lhs, const float factor) {
	return factor * lhs;
}
inline Vector operator/(const Vector& lhs, const float factor) {
	return Vector(lhs[0] / factor, lhs[1] / factor, lhs[2] / factor);
}
inline std::ostream& operator<<(std::ostream& out, const Vector& rhs) {
	return out << "x: " << rhs[0] << "\ty: " << rhs[1] << "\tz: " << rhs[2];
}

#endif /* VECTOR_H_ */
