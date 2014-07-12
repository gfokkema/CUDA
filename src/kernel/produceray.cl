typedef struct cl_camera {
	int width, height;
	float4 pos;
	float4 dir;
	float4 up;
	float4 right;
} cl_camera;

__kernel void produceray(__global float4* output, cl_camera cam) {
	int yi = get_global_id(0);
	int offset = yi * cam.width;
	
	float invwidth = 1.f / cam.width;
	float invheight = 1.f / cam.height;
	
	for (int xi = 0; xi < cam.width; xi++) {
		float x = (xi + .5) * invwidth - 0.5;
		float y = (yi + .5) * invheight - 0.5;
		
		output[offset + xi] = normalize(x * cam.right + y * cam.up + cam.dir);
	}
}
