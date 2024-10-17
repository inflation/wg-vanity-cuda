cmake_minimum_required(VERSION 3.18)

project(curve25519-cuda LANGUAGES C CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)
add_library(curve25519-cuda ${SOURCES})
target_link_libraries(curve25519-cuda CUDA::curand)
