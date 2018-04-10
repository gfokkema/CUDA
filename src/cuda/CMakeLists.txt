set (kernel_sources
	produce.cu
	pathtrace.cu
)

add_library (kernel ${kernel_sources})
