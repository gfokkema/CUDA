set (device_sources
	cudadevice.cpp
	device.cpp
)
cuda_add_library (device ${device_sources})
