set (opencl_sources
	opencl.cpp
	produceray.cpp
	traceray.cpp
)
add_library (opencl ${opencl_sources})
