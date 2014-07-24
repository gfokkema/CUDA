# The kernel file string (which is just kernel.cl)
set (kernel_file_str kernel.cl)

# The openclc compiler will only output bitcode files if the target directory exists
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/")
	EXEC_PROGRAM(mkdir ${CMAKE_CURRENT_BINARY_DIR}/OpenCL/)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.c")
	EXEC_PROGRAM(rm "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.c")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.h")
	EXEC_PROGRAM(rm "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.h")
endif()

# The header file is also derived. 
set (kernel_derived_gcl
	${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.c
)

# This list will be filled with the LLVM bitcode files for each architecture, specified by kernel_archs
set (kernel_bitcode_files)

# Architectures for which we will compile
set (kernel_archs
	gpu_64
	x86_64
	i386
	gpu_32
)

add_library(
	kernel
	STATIC
	EXCLUDE_FROM_ALL
	${kernel_derived_gcl}
)

set_target_properties(
	kernel
	PROPERTIES
	EXTERNAL_OBJECT true
	LINKER_LANGUAGE C
	ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# For each architecture, we generate LLVM bitcode. This will be generated before the generated GCL files
foreach(arch ${kernel_archs})
	set(bitcode_file ${CMAKE_CURRENT_BINARY_DIR}/OpenCL/${kernel_file_str}.${arch}.bc)
	list(APPEND kernel_bitcode_files ${bitcode_file})
	add_custom_command(
		OUTPUT ${bitcode_file}
		PRE_BUILD
		COMMAND /System/Library/Frameworks/OpenCL.framework/Libraries/openclc -D__APPLE_KERNEL_COMPILE__ -I../util -x cl -cl-std=CL1.1 -Os -arch ${arch} -emit-llvm -c ${kernel_file_str} -o ${bitcode_file}
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)
endforeach()

# The derived GCL files (kernel.cl.c and kernel.cl.h)
add_custom_command(
	OUTPUT ${kernel_derived_gcl}
	COMMAND /System/Library/Frameworks/OpenCL.framework/Libraries/openclc -D__APPLE_KERNEL_COMPILE__ -I../util -x cl -cl-std=CL1.1 -cl-auto-vectorize-enable -gcl-bc-dir kernel/OpenCL -emit-gcl ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl
	DEPENDS ${kernel_bitcode_files}
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
