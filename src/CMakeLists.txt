cmake_minimum_required (VERSION 2.6)

project (cuda-testing)

add_definitions(-std=c++11)

#add_subdirectory (cuda)
add_subdirectory (shapes)

add_executable (main main.cpp)
add_executable (raytracer scene.cpp raytracer.cpp)
target_link_libraries (raytracer shapes)
