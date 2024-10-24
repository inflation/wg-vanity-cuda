cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_CUDA_COMPILER_LAUNCHER sccache)


include(FetchContent)
FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
)
FetchContent_Declare(aegis
        GIT_REPOSITORY https://github.com/jedisct1/libaegis
        GIT_TAG 0.1.23
)
FetchContent_MakeAvailable(fmt aegis)

project(curve25519-cuda LANGUAGES CXX CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_libraries(fmt::fmt)

add_compile_definitions(FMT_UNICODE=0)

file(GLOB_RECURSE SOURCES src/*.cu)
message(STATUS "Sources: ${SOURCES}")

add_executable(vanity ${SOURCES})
target_link_libraries(vanity fmt::fmt aegis::aegis "$<${WIN32}:bcrypt>")
target_include_directories(vanity PRIVATE ${aegis_SOURCE_DIR}/src/include)
