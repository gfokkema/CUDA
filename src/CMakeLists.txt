cmake_minimum_required (VERSION 2.6)

project (cuda-testing)

find_package (CUDA REQUIRED)

add_definitions(-std=c++11)
list (APPEND CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20")

cuda_add_executable (add add.cu)
cuda_add_executable (saxpy saxpy.cu)
