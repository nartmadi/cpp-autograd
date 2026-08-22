cmake_minimum_required(VERSION 3.20)

project(autograd_cpp LANGUAGES CXX) # Append CUDA later

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(autograd
	src/value.cpp
)

target_include_directories(autograd
	PUBLIC ${PROJECT_SOURCE_DIR}/include
)

add_executable(demo
	src/main.cpp
)

target_link_libraries(demo
	PRIVATE autograd
)

