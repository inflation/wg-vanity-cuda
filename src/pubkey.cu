#include <cassert>
#include <cuda_runtime.h>

#include "base64_encode.h"
#include "pubkey.h"

__global__ void pubkey(const uint8_t *prv_keys, uint8_t *encoded_pub_keys,
                       uint8_t *encoded_prv_keys) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;

  auto e = prv_keys + idx * key_len;
  auto e_pubk = encoded_pub_keys + idx * encoded_len;
  auto e_prvk = encoded_prv_keys + idx * encoded_len;

  curved25519_key pk;
  curved25519_scalarmult_basepoint(pk, e);
  base64::encode(pk, e_pubk);
  base64::encode(e, e_prvk);
}

int main() {
  constexpr auto num_blocks = 1;
  constexpr auto num_threads = 2;
  constexpr auto num_keys = num_blocks * num_threads;
  constexpr auto mem_keys = num_keys * key_len;
  constexpr auto mem_encoded_keys = num_keys * encoded_len;

  uint8_t *h_es, *h_e_pubks, *h_e_prvks;
  cudaMallocHost(&h_es, mem_keys);
  cudaMallocHost(&h_e_pubks, mem_encoded_keys);
  cudaMallocHost(&h_e_prvks, mem_encoded_keys);
  for (int i = 0; i < num_keys; i++) {
    memset(h_es + i * key_len, 0, key_len);
  }

  uint8_t *d_es, *d_e_pubks, *d_e_prvks;
  cudaMalloc(&d_es, mem_keys);
  cudaMalloc(&d_e_pubks, mem_encoded_keys);
  cudaMalloc(&d_e_prvks, mem_encoded_keys);
  cudaMemcpy(d_es, h_es, mem_keys, cudaMemcpyHostToDevice);

  pubkey<<<num_blocks, num_threads>>>(d_es, d_e_pubks, d_e_prvks);

  cudaMemcpy(h_e_pubks, d_e_pubks, mem_encoded_keys, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_e_prvks, d_e_prvks, mem_encoded_keys, cudaMemcpyDeviceToHost);

  assert(memcmp(h_e_pubks, h_e_pubks + encoded_len, encoded_len) == 0);
  assert(memcmp(h_e_prvks, h_e_prvks + encoded_len, encoded_len) == 0);

  cudaDeviceReset();

  return 0;
}