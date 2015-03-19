set (kernel_sources
	produce.cu
	pathtrace.cu
)

cuda_add_library (kernel ${kernel_sources})