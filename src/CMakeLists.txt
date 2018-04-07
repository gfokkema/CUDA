cmake_minimum_required (VERSION 2.6)
project (RT-Raytracing)

add_definitions(-std=c++14 -g -O3)
list (APPEND CUDA_NVCC_FLAGS "-std=c++14 -g -O3 -arch=compute_50; -code=sm_50")

find_package (CUDA)
find_package (OpenGL REQUIRED)
find_package (PkgConfig REQUIRED)
pkg_search_module (GLFW REQUIRED glfw3)
pkg_search_module (GLEW REQUIRED glew)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory (cuda)
add_subdirectory (device)
add_subdirectory (util)
cuda_add_executable   (raytracer window.cpp scene.cpp main.cpp)
target_link_libraries (raytracer device kernel util ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${CUDA_curand_LIBRARY})
