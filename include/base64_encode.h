#include <cuda_runtime.h>

#include "common.h"

using namespace curve25519;

namespace base64 {

__host__ __device__ __noinline__ void encode(const curved25519_key key,
                                             encoded_key out) {
  static const char *base64_table =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  for (int i = 0, j = 0; i < key_len; i += 3, j += 4) {
    uint32_t octet_a = key[i];
    uint32_t octet_b = key[i + 1];
    uint32_t octet_c = key[i + 2];

    uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;

    out[j] = base64_table[(triple >> 18) & 0x3F];
    out[j + 1] = base64_table[(triple >> 12) & 0x3F];
    out[j + 2] = base64_table[(triple >> 6) & 0x3F];
    out[j + 3] = base64_table[triple & 0x3F];
  }

  out[encoded_len - 1] = '=';
}

} // namespace base64