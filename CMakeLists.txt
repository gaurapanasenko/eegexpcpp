cmake_minimum_required(VERSION 3.18)
project(eegexpcpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
endif()

find_package(Torch REQUIRED)

add_executable(eegexpcpp src/main.cpp)
target_link_libraries(eegexpcpp PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(eegexpcpp PRIVATE ${TORCH_CXX_FLAGS})
set_property(TARGET eegexpcpp PROPERTY CXX_STANDARD 20)
