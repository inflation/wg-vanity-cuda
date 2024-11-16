cmake_minimum_required(VERSION 3.18)

project(curve25519-cuda LANGUAGES CXX CUDA)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_COMPILER_LAUNCHER sccache)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_COMPILER_LAUNCHER sccache)


include(FetchContent)
FetchContent_Declare(aegis
        GIT_REPOSITORY https://github.com/jedisct1/libaegis
        GIT_TAG 0.1.23
)
FetchContent_MakeAvailable(aegis)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES src/*.cu)
message(STATUS "Sources: ${SOURCES}")

add_library(vanity ${SOURCES})
target_include_directories(vanity PRIVATE ${aegis_SOURCE_DIR}/src/include)

add_executable(vanity-cli src/main.cpp)
target_link_libraries(vanity-cli vanity aegis::aegis "$<${WIN32}:bcrypt>")
target_include_directories(vanity-cli PRIVATE ${aegis_SOURCE_DIR}/src/include)
