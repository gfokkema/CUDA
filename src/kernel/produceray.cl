typedef struct cl_camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
} cl_camera;

__kernel void produceray(__global float* output, cl_camera cam) {
	printf("debug %d\n", cam.width);
}
