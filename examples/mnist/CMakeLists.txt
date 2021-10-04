cmake_minimum_required(VERSION 3.16)

project(mnist CXX)

add_executable(mnist
	main.cpp
)

set_property(TARGET mnist PROPERTY CXX_STANDARD 20)

target_link_libraries(mnist libonnx)

