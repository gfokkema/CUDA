cmake_minimum_required (VERSION 2.6)

project (RT-Raytracing)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake/modules" ${CMAKE_MODULE_PATH})

find_package (GLEW REQUIRED)
find_package (glfw REQUIRED)
find_package (OpenGL REQUIRED)
find_package (CUDA)
IF (CUDA_FOUND)
	list (APPEND CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20")
	add_subdirectory (cuda)
	add_definitions(-DCUDA_FOUND)
ELSE (CUDA_FOUND) #CUDA not found, look for OpenCL
	find_package (OpenCL REQUIRED)
	include_directories (${OPENCL_INCLUDE_DIR})
ENDIF (CUDA_FOUND)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	include_directories(/opt/X11/include)
	include_directories(/opt/local/include)
	link_directories(/opt/X11/lib)
	link_directories(/opt/local/lib)
	add_definitions(-w)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_definitions(-std=c++11)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory (devices)
add_subdirectory (util)
add_executable (raytracer scene.cpp window.cpp)
target_link_libraries (raytracer devices util ${OPENGL_LIBRARIES} ${OPENCL_LIBRARIES} ${GLFW_LIBRARY} ${GLEW_LIBRARY})

IF (CUDA_FOUND)
	target_link_libraries (raytracer kernel)
ENDIF (CUDA_FOUND)
