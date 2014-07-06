cmake_minimum_required (VERSION 2.6)

project (cuda-testing)

find_package (CUDA REQUIRED)

add_definitions(-std=c++11)
list (APPEND CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20")

add_subdirectory (cuda)

add_executable (main main.cpp)
add_executable (raytracer scene.cpp raytracer.cpp)
