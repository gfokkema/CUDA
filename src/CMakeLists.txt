cmake_minimum_required (VERSION 2.6)

project (RT-Raytracing)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake/modules" ${CMAKE_MODULE_PATH})

add_definitions (-std=c++11 -O3)

find_package (GLEW REQUIRED)
find_package (glfw REQUIRED)
find_package (OpenGL REQUIRED)
find_package (OpenCL REQUIRED)
#find_package (CUDA)
#include_directories(/opt/cuda/include)
#IF (CUDA_FOUND)
#	list (APPEND CUDA_NVCC_FLAGS "-O3 -arch=compute_20; -code=sm_20")
#	add_subdirectory (cuda)
#	add_definitions(-DCUDA_FOUND)
#ENDIF (CUDA_FOUND) #CUDA not found, look for OpenCL


include_directories (${CMAKE_SOURCE_DIR} ${OPENCL_INCLUDE_DIR})
set (libraries
	devices util
	${OPENGL_LIBRARIES}
	${OPENCL_LIBRARIES}
	${GLFW_LIBRARY}
	${GLEW_LIBRARY}
)

IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_definitions(-Wall -fblocks)

	include_directories (/opt/X11/include /opt/local/include)
	link_directories (/opt/X11/lib /opt/local/lib)
	add_subdirectory (kernel)
	add_dependencies (devices kernel)
	list (APPEND libraries kernel)
ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
IF (CUDA_FOUND)
	list (APPEND libraries cuda_kernel)
ENDIF (CUDA_FOUND)

add_subdirectory (devices)
add_subdirectory (util)
add_executable (raytracer scene.cpp window.cpp session.cpp)

target_link_libraries (raytracer ${libraries})
