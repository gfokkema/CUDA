cmake_minimum_required (VERSION 2.6)

project (RT-Raytracing)

find_package (GLEW REQUIRED)
find_package (glfw REQUIRED)
find_package (OpenGL REQUIRED)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	include_directories(/opt/X11/include)
	include_directories(/opt/local/include)
	link_directories(/opt/X11/lib)
	link_directories(/opt/local/lib)
	add_definitions(-w)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_definitions(-std=c++11)

include_directories("include")

add_subdirectory (shapes)
add_subdirectory (util)
add_executable (main main.cpp)
add_executable (raytracer scene.cpp window.cpp)
target_link_libraries (raytracer util shapes ${OPENGL_LIBRARIES} ${GLFW_LIBRARY} ${GLEW_LIBRARY} OpenCL)
