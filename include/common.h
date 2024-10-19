#pragma once

#include <cstdint>

#define mul32x32_64(a, b) ((uint64_t)(a) * (b))

#ifdef _MSC_VER
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT static const
#endif

namespace curve25519 {

typedef uint32_t bignum25519[10];
typedef uint32_t bignum25519align16[12];

/*
 * Arithmetic on the twisted Edwards curve -x^2 + y^2 = 1 + dx^2y^2
 * with d = -(121665/121666) =
 * 37095705934669439343138083508754565189542113879843219016388785533085940283555
 * Base point:
 * (15112221349535400772501151409588531511454012693041857206046113283949847762202,46316835694926478169428394003475163141307993866256225615783033603165251855960);
 */

typedef struct ge25519_t {
  bignum25519 x, y, z, t;
} ge25519;

typedef struct ge25519_p1p1_t {
  bignum25519 x, y, z, t;
} ge25519_p1p1;

typedef struct ge25519_niels_t {
  bignum25519 ysubx, xaddy, t2d;
} ge25519_niels;

typedef struct ge25519_pniels_t {
  bignum25519 ysubx, xaddy, z, t2d;
} ge25519_pniels;

constexpr auto key_len = 32;
constexpr auto encoded_len = (key_len + 2) / 3 * 4;

typedef uint8_t curved25519_key[key_len];
typedef uint8_t encoded_key[encoded_len];

} // namespace curve25519