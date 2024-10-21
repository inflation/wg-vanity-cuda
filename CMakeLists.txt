cmake_minimum_required(VERSION 3.18)

project(curve25519-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_COMPILER_LAUNCHER sccache)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(vanity src/common.cc src/base64.cu src/pubkey.cu src/vanity.cu)
target_link_libraries(vanity CUDA::curand)