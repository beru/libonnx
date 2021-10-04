cmake_minimum_required(VERSION 3.16)

project(benchmark CXX)

add_executable(benchmark
	main.cpp
)

set_property(TARGET benchmark PROPERTY CXX_STANDARD 20)

target_link_libraries(benchmark libonnx)

