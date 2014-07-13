#ifndef SHAPE_H_
#define SHAPE_H_

#include <CL/cl.hpp>

enum type {
	SPHERE,
	PLANE,
	TRIANGLE
};

typedef struct cl_shape {
	union {
		// SPHERE
		struct {
			cl_float4 origin;	// offset 0
			cl_float radius;	// offset 16
		} sphere;
		// PLANE
		struct {
			cl_float4 origin;	// offset 0
			cl_float4 normal;	// offset 16
		} plane;
		// TRIANGLE
		struct {
			cl_float4 v1;		// offset 0
			cl_float4 v2;		// offset 16
			cl_float4 v3;		// offset 32
		} triangle;
	};

	int type;					// offset 48
} cl_shape;

#endif /* SHAPE_H_ */
