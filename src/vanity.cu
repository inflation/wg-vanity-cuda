#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "vanity.h"

__global__ void test_vanity() {
  vanity(reinterpret_cast<const uint8_t *>("asuna"), 5);
}

int main() {
  int blockSize, minGridSize, maxActiveBlocks, gridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, test_vanity);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, test_vanity,
                                                blockSize, 0);
  printf("blockSize: %d, minGridSize: %d, maxActiveBlocks: %d\n", blockSize,
         minGridSize, maxActiveBlocks);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = ((float)maxActiveBlocks * blockSize / props.warpSize) /
                    ((float)props.maxThreadsPerMultiProcessor / props.warpSize);

  gridSize = minGridSize * 1024;
  test_vanity<<<gridSize, blockSize>>>();
  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", blockSize,
         occupancy);

  cudaDeviceReset();
  return 0;
}