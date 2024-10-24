#pragma once

#include "common.h"

using namespace curve25519;

namespace base64 {

__device__ void encode(const curved25519_key key, encoded_key out);
__device__ void decode(const encoded_key key, curved25519_key out);

} // namespace base64