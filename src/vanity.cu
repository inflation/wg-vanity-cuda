#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "base64.h"
#include "common.h"
#include "pubkey.h"
#include "random.h"
#include "vanity.h"

using namespace curve25519;

__constant__ uint8_t vanity_str[KEY_LEN_BASE64];
__device__ uint32_t foundKeys = 1;

__global__ void vanity(size_t vanity_len, const uint8_t *__restrict d_keys,
                       size_t key_mem_size) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  for (auto i = idx; i + 32 < key_mem_size; i += gridDim.x * blockDim.x) {
    auto e = d_keys + i;
    curved25519_key pk;
    encoded_key e_pubk, e_prvk;
    pubkey(pk, e);
    base64::encode(pk, e_pubk);

    bool found = true;
    for (int j = 0; j < vanity_len; j++) {
      if (e_pubk[j] != vanity_str[j]) {
        found = false;
        break;
      }
    }

    if (found) {
      auto o_idx = atomicAdd(&foundKeys, 1);
      base64::encode(e, e_prvk);

      printf("found key %2d: pub %.44s, prv %.44s\n", o_idx,
             reinterpret_cast<const char *>(e_pubk),
             reinterpret_cast<const char *>(e_prvk));
      return;
    }
  }
}

void find_pubkey(char *str, int len, int rounds, int mem, int threads) {
  int blockSize, minGridSize;
  checkCudaError(
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity));
  checkCudaError(cudaMemcpyToSymbol(vanity_str, str, len));

  std::vector<std::thread> workers;

  for (auto i = 0; i < threads; i++) {
    workers.emplace_back([=]() {
      randState state(0, mem * 1_MB);

      for (auto j = 0; j < rounds; j++) {
        generate(state);
        vanity<<<minGridSize, blockSize, 0, state.stream>>>(len, state.d_keys,
                                                            state.mem_size);
        checkCudaError(cudaStreamSynchronize(state.stream));
      }
    });
  }

  for (auto &t : workers) {
    t.join();
  }

  checkCudaError(cudaDeviceSynchronize());
  checkCudaError(cudaDeviceReset());
}