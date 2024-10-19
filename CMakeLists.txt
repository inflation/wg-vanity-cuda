cmake_minimum_required(VERSION 3.18)

project(curve25519-cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)

# file(GLOB CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)
# add_executable(curve25519-cuda ${CUDA_SOURCES})
# target_link_libraries(curve25519-cuda CUDA::curand)

add_executable(vanity src/vanity.cu)
target_link_libraries(vanity CUDA::curand)