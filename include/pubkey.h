#pragma once

#include "common.h"

namespace curve25519 {

__device__ void pubkey(curved25519_key pk, const curved25519_key e);

}  // namespace curve25519