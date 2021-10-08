cmake_minimum_required(VERSION 3.16)

project(mnist CXX)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(mnist
	main.cpp
)

set_property(TARGET mnist PROPERTY CXX_STANDARD 20)

target_link_libraries(mnist libonnx)

