cmake_minimum_required(VERSION 3.16)

project(hello CXX)

add_executable(hello
	main.cpp
)

set_property(TARGET hello PROPERTY CXX_STANDARD 20)

target_link_libraries(hello libonnx)
