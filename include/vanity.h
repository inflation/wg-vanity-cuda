#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "base64_encode.h"
#include "common.h"
#include "pubkey.h"

namespace curve25519 {

constexpr int ITER_PER_THREADS = 1 << 10;

__device__ int foundKeys = 0;

__device__ void vanity(const uint8_t *vanity, size_t len) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(clock64(), idx, 0, &state);

  auto rand = curand4(&state);
  auto e = reinterpret_cast<uint8_t *>(&rand);

  curve25519::encoded_key e_pubk, e_prvk;

  curved25519_key pk;

  for (int i = 0; i < ITER_PER_THREADS; i++) {
    curved25519_scalarmult_basepoint(pk, e);
    base64::encode(pk, e_pubk);

    bool found = true;
    for (int j = 0; j < len; j++) {
      if (e_pubk[j] != vanity[j]) {
        found = false;
      }
    }
    if (found) {
      auto o_idx = atomicAdd(&foundKeys, 1);
      base64::encode(e, e_prvk);

      printf("found key %d: %.44s\n", o_idx, e_pubk);
    }
  }
}

} // namespace curve25519