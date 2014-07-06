find_package (CUDA REQUIRED)

list (APPEND CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20")

cuda_add_executable (add add.cu)
cuda_add_executable (saxpy saxpy.cu)
