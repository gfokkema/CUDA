cmake_minimum_required (VERSION 2.6)
project (RT-Raytracing)

add_definitions(-std=c++11 -O3)
list (APPEND CUDA_NVCC_FLAGS "-O3 -arch=compute_50; -code=sm_50")

find_package (CUDA)
find_package (OpenGL REQUIRED)
find_package (PkgConfig REQUIRED)
pkg_search_module (GLFW REQUIRED glfw3)
pkg_search_module (GLEW REQUIRED glew)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory (cuda)
add_subdirectory (util)
add_executable (raytracer scene.cpp window.cpp)
target_link_libraries (raytracer kernel util ${OPENGL_LIBRARIES} ${GLFW_LIBRARY} ${GLEW_LIBRARY})
