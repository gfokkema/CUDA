cmake_minimum_required (VERSION 2.6)
project (RT-Raytracing)

add_definitions (-std=c++11 -O3)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake/modules" ${CMAKE_MODULE_PATH})

find_package (OpenCL REQUIRED)
find_package (OpenGL REQUIRED)
find_package (PkgConfig REQUIRED)
pkg_search_module (GLFW REQUIRED glfw3)
pkg_search_module (GLEW REQUIRED glew)

include_directories (${CMAKE_SOURCE_DIR} ${OPENCL_INCLUDE_DIR})
set (libraries
	devices util
	${OPENGL_LIBRARIES}
	${OPENCL_LIBRARIES}
	${GLFW_LIBRARY}
	${GLEW_LIBRARY}
)

IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_definitions (-Wall -fblocks)
	include_directories (/opt/X11/include /opt/local/include)
	link_directories (/opt/X11/lib /opt/local/lib)
	add_subdirectory (kernel)
ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_subdirectory (devices)
add_subdirectory (util)
add_executable (raytracer scene.cpp window.cpp session.cpp)

IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_dependencies (devices kernel)
	list (APPEND libraries kernel)
ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

target_link_libraries (raytracer ${libraries})

