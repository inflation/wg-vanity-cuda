#include "pubkey.h"
#include "ed25519-donna.h"

namespace curve25519 {

__device__ void pubkey(curved25519_key pk, const curved25519_key e) {
  curved25519_key ec;
  bignum256modm s;
  bignum25519 __align__(16) yplusz, zminusy;
  ge25519 __align__(16) p;

  /* clamp */
  // for (auto i = 0; i < KEY_LEN; i++)
    // ec[i] = e[i];
  memcpy(ec, e, KEY_LEN);
  ec[0] &= 248;
  ec[31] &= 127;
  ec[31] |= 64;

  expand_raw256_modm(s, ec);

  /* scalar * basepoint */
  ge25519_scalarmult_base_niels(&p, ge25519_niels_base_multiples, s);

  /* u = (y + z) / (z - y) */
  curve25519_add(yplusz, p.y, p.z);
  curve25519_sub(zminusy, p.z, p.y);
  curve25519_recip(zminusy, zminusy);
  curve25519_mul(yplusz, yplusz, zminusy);
  curve25519_contract(pk, yplusz);
}

} // namespace curve25519