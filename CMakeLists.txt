cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(demo)
set(CMAKE_PREFIX_PATH "/root/torch_test/libtorch") 

find_package(Torch REQUIRED)
add_executable(demo "main.cc")
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
