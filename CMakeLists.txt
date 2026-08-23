cmake_minimum_required(VERSION 3.20)

project(scalar_autograd LANGUAGES CXX)

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

enable_testing()

add_executable(value_tests
	tests/value_tests.cpp
)

add_executable(backward_tests
	tests/backward_tests.cpp
)

target_link_libraries(value_tests
	PRIVATE autograd
)

target_link_libraries(backward_tests
	PRIVATE autograd
)

add_test(NAME value_tests COMMAND value_tests)
add_test(NAME backward_tests COMMAND backward_tests)
