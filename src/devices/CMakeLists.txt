set (devices_sources
	cpudevice.cpp
	opencl.cpp
)
IF (CUDA_FOUND)
	list(APPEND devices_sources cudadevice.cpp)
	cuda_add_library(cudadevice cudadevice.cpp)
ENDIF (CUDA_FOUND)

add_library (devices ${devices_sources} cudadevice)
