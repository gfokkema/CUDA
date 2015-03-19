set (kernel_sources
	host_kernels.cu
	produce.cu
	trace.cu
	pathtrace.cu
)

cuda_add_library (kernel ${kernel_sources})