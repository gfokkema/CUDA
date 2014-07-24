if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/")
	EXEC_PROGRAM(mkdir ${CMAKE_CURRENT_BINARY_DIR}/OpenCL/)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.c")
	EXEC_PROGRAM(rm "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.c")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.h")
	EXEC_PROGRAM(rm "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.cl.h")
endif()

set (kernel_file_str kernel.cl)
# set (kernel_object_file 
	# ${CMAKE_CURRENT_BINARY_DIR}/${kernel_file_str}.o
# )
set (kernel_derived_gcl
	${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_str}.c
)
set (kernel_bitcode_files
	#${CMAKE_CURRENT_BINARY_DIR}/OpenCL/${kernel_file_str}.gpu_64.bc
	#${CMAKE_CURRENT_BINARY_DIR}/OpenCL/${kernel_file_str}.x86_64.bc
	#${CMAKE_CURRENT_BINARY_DIR}/OpenCL/${kernel_file_str}.i386.bc
	#${CMAKE_CURRENT_BINARY_DIR}/OpenCL/${kernel_file_str}.gpu_32.bc
)
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

# set_source_files_properties(
	# ${kernel_object_file}
	# PROPERTIES
	# EXTERNAL_OBJECT	true # Indicate that it's an object file
	# GENERATED	true # Object file may not be present at build time
# )

set_target_properties(
	kernel
	PROPERTIES
	EXTERNAL_OBJECT true
	LINKER_LANGUAGE C # Is this the right one?
	ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

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

add_custom_command(
	OUTPUT ${kernel_derived_gcl}
	COMMAND /System/Library/Frameworks/OpenCL.framework/Libraries/openclc -D__APPLE_KERNEL_COMPILE__ -I../util -x cl -cl-std=CL1.1 -cl-auto-vectorize-enable -gcl-bc-dir kernel/OpenCL -emit-gcl ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl
	DEPENDS ${kernel_bitcode_files}
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
