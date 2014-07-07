cmake_minimum_required (VERSION 2.6)

project (cuda-testing)

find_package (GLEW REQUIRED)
find_package (OpenGL REQUIRED)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	include_directories(/opt/X11/include)
	link_directories(/opt/X11/lib)
	add_definitions(-w)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_definitions(-std=c++11)

add_subdirectory (shapes)

add_executable (main main.cpp)
add_executable (raytracer scene.cpp window.cpp)
target_link_libraries (raytracer shapes ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
