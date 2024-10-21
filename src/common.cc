#include <format>
#include <iostream>

#include <cuda_runtime.h>

#include "common.h"

void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cout << std::format("CUDA error at {} {}: {}\n", __FILE__, __LINE__,
                             cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void checkLastCudaError() {
  checkCudaError(cudaGetLastError());
}