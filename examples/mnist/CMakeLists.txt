cmake_minimum_required(VERSION 3.16)

project(mnist CXX)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(mnist
	main.cpp
	SDL2_framerate.c
	SDL2_gfxPrimitives.c
	SDL2_imageFilter.c
	SDL2_rotozoom.c
)

set_property(TARGET mnist PROPERTY CXX_STANDARD 20)

target_link_libraries(mnist libonnx ${SDL2_LIBRARIES})

