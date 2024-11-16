#pragma once

#include <array>
#ifdef _WIN32
#include <windows.h>
#endif

#include <aegis.h>
#include <aegis128x4.h>
#include <cuda_runtime.h>

#include "common.h"

struct randState {
  uint32_t thread;
  size_t mem_size;

  cudaStream_t stream{};
  uint8_t *h_keys = nullptr;
  uint8_t *d_keys = nullptr;
  randState(const uint32_t t, const size_t m)
      : thread(t), mem_size(m) {
    checkCudaError(cudaStreamCreate(&stream));
    checkCudaError(cudaMallocHost(&h_keys, mem_size));
    checkCudaError(cudaMallocAsync(&d_keys, mem_size, stream));
  }
};

template <auto seed_size = 16>
static auto generate_seed() {
  std::array<uint8_t, seed_size> seed;
#ifdef _WIN32
  const auto status = BCryptGenRandom(nullptr, seed.data(), seed_size,
                                      BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (!BCRYPT_SUCCESS(status)) {
    std::cerr << "BCryptGenRandom failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  return seed;
}
#else
  const auto fp = fopen("/dev/urandom", "r");
  fread(&seed.data(), 1, seed_size, fp);
  fclose(fp);
  return seed;
}
#endif

static void init() {
  aegis_init();
}

static void generate(const randState &state) {
  const auto seed = generate_seed();
  aegis128x4_stream(state.h_keys, state.mem_size, nullptr, seed.data());

  checkCudaError(cudaMemcpyAsync(state.d_keys, state.h_keys, state.mem_size,
                                 cudaMemcpyHostToDevice, state.stream));
}

