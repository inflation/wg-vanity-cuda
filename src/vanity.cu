#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "common.h"
#include "vanity.h"

__device__ uint8_t vanity_str[KEY_LEN_BASE64];

__global__ void test_vanity(size_t len) {
  vanity(vanity_str, len);
}

int main(int argc, char** argv) {
  char* str = argv[1];
  size_t len = strlen(str);
  printf("String: %s, Length: %zu\n", str, len);

  checkCudaError(cudaMemcpyToSymbol(vanity_str, str, len));

  int blockSize, minGridSize, maxActiveBlocks, gridSize;
  checkCudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                                    test_vanity));
  checkCudaError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, test_vanity, blockSize, 0));
  printf("blockSize: %d, minGridSize: %d, maxActiveBlocks: %d\n", blockSize,
         minGridSize, maxActiveBlocks);

  int device;
  cudaDeviceProp props;
  checkCudaError(cudaGetDevice(&device));
  checkCudaError(cudaGetDeviceProperties(&props, device));

  float occupancy = ((float)maxActiveBlocks * blockSize / props.warpSize) /
                    ((float)props.maxThreadsPerMultiProcessor / props.warpSize);

  gridSize = minGridSize * 1024;
  test_vanity<<<gridSize, blockSize>>>(len);
  checkLastCudaError();

  std::cout << "Launched blocks of size: " << blockSize
            << ". Theoretical occupancy: " << occupancy << std::endl;

  checkCudaError(cudaDeviceReset());
  return 0;
}