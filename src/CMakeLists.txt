cmake_minimum_required(VERSION 3.16)

project(libonnx)

include_directories(
	./
)

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR} *.h *.c *.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/.*" )

add_library(libonnx
	${SRC_FILES}
)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

set_property(TARGET libonnx PROPERTY CXX_STANDARD 20)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	target_compile_options(libonnx PRIVATE /MP)
	target_compile_options(libonnx PRIVATE /Zc:preprocessor)
	
	set_property(TARGET libonnx PROPERTY
		DIRECTORY PROPERTY VS_STARTUP_PROJECT
		)
	
	set_property(TARGET libonnx PROPERTY
		MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
		)
	
endif()

