#define EPSILON 1e-4

typedef struct camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
} camera;

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

extern "C" int cudaproduceray(camera cam, float4*& raydirs);
