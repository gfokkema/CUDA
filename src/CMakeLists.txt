cmake_minimum_required (VERSION 2.6)

project (RT-Raytracing)

find_package (GLEW REQUIRED)
find_package (glfw REQUIRED)
find_package (OpenGL REQUIRED)
find_package (CUDA)
IF (CUDA_FOUND)
	list (APPEND CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20")
	add_subdirectory (cuda)
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
target_link_libraries (raytracer devices util ${OPENGL_LIBRARIES} ${GLFW_LIBRARY} ${GLEW_LIBRARY} OpenCL)

IF (CUDA_FOUND)
	target_link_libraries (raytracer kernel)
ENDIF (CUDA_FOUND)
