#include <cstdint>

#include <cuda_runtime.h>

#include "base64.h"

using namespace curve25519;

namespace base64 {


__constant__ uint8_t base64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                      "abcdefghijklmnopqrstuvwxyz"
                                      "0123456789+/";
__constant__ uint8_t from_base64[] = {
    62,  255, 255, 255, 63,  52,  53, 54, 55, 56, 57, 58, 59, 60, 61, 255,
    255, 255, 255, 255, 255, 255, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    10,  11,  12,  13,  14,  15,  16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    255, 255, 255, 255, 255, 255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36,  37,  38,  39,  40,  41,  42, 43, 44, 45, 46, 47, 48, 49, 50, 51};

__device__ void encode(const curved25519_key key, encoded_key out) {
  for (int i = 0, j = 0; i < KEY_LEN;) {
    uint32_t octet_a = key[i++];
    uint32_t octet_b = key[i++];
    uint32_t octet_c = i < KEY_LEN ? key[i++] : 0;

    uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;

    out[j++] = base64_table[(triple >> 18) & 0x3F];
    out[j++] = base64_table[(triple >> 12) & 0x3F];
    out[j++] = base64_table[(triple >> 6) & 0x3F];
    out[j++] = base64_table[triple & 0x3F];
  }

  out[KEY_LEN_BASE64 - 1] = '=';
}

__device__ void decode(const encoded_key key, curved25519_key out) {
  for (int i = 0, j = 0; i < KEY_LEN_BASE64;) {
    uint32_t sextet_a = from_base64[key[i++] - '+'];
    uint32_t sextet_b = from_base64[key[i++] - '+'];
    uint32_t sextet_c = from_base64[key[i++] - '+'];
    uint32_t sextet_d = from_base64[key[i++] - '+'];

    uint32_t triple =
        (sextet_a << 18) | (sextet_b << 12) | (sextet_c << 6) | sextet_d;

    out[j++] = (triple >> 16) & 0xFF;
    out[j++] = (triple >> 8) & 0xFF;
    out[j++] = triple & 0xFF;
  }
}


}  // namespace base64
