#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "base64.h"
#include "common.h"
#include "pubkey.h"
#include "random.h"

using namespace curve25519;

__constant__ uint8_t vanity_str[KEY_LEN_BASE64];
__device__ uint32_t foundKeys = 0;

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

int main(int argc, char **argv) {
  init();

  if (argc < 2) {
    fmt::println(stderr, "Usage: {} <vanity> [THREADS] [ROUNDS]", argv[0]);
    exit(EXIT_FAILURE);
  }

  char *str = argv[1];
  size_t len = strlen(str);
  printf("String: %s, Length: %zu\n", str, len);

  checkCudaError(cudaMemcpyToSymbol(vanity_str, str, len));

  int blockSize, minGridSize;
  checkCudaError(
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity));
  fmt::println("blockSize: {}, minGridSize: {}", blockSize, minGridSize);

  uint32_t rounds = argc == 3 ? atoi(argv[2]) : 10;
  uint32_t threads =
      argc == 4 ? atoi(argv[4]) : std::thread::hardware_concurrency();

  std::vector<std::thread> workers;

  for (auto i = 0; i < threads; i++) {
    workers.emplace_back([=]() {
      randState state(0, 100_MB);

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
  return 0;
}