cmake_minimum_required(VERSION 3.16)

project(tests CXX)

add_executable(tests
	main.cpp
)

set_property(TARGET tests PROPERTY CXX_STANDARD 20)

target_link_libraries(tests libonnx)

