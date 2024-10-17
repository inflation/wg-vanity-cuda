#include <cstdint>

#include <cuda_runtime.h>

#define mul32x32_64(a, b) ((uint64_t)(a) * (uint64_t)(b))
#define ALIGN(x) __attribute__((aligned(x)))

__host__ __device__ static inline uint32_t U8TO32_LE(const unsigned char *p) {
  return (((uint32_t)(p[0])) | ((uint32_t)(p[1]) << 8) |
          ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24));
}

__host__ __device__ static inline void U32TO8_LE(unsigned char *p,
                                                 const uint32_t v) {
  p[0] = (unsigned char)(v);
  p[1] = (unsigned char)(v >> 8);
  p[2] = (unsigned char)(v >> 16);
  p[3] = (unsigned char)(v >> 24);
}