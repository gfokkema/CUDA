#ifndef SHAPE_H_
#define SHAPE_H_

#include "util/vector.h"

enum type {
	SPHERE,
	PLANE,
	TRIANGLE
};

typedef struct shape {
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

	int type;					// offset 48
} shape;

#endif /* SHAPE_H_ */
