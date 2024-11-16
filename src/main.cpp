#include <cstdio>
#include <print>
#include <thread>

#include <vanity.h>
#include <aegis.h>

int main(int argc, char **argv) {
  aegis_init();

  if (argc < 2) {
    std::println(stderr, "Usage: {} <vanity> [ROUNDS] [PER_THREAD_MEM] [THREADS]", argv[0]);
    exit(EXIT_FAILURE);
  }

  char *str = argv[1];
  size_t len = strlen(str);
  printf("String: %s, Length: %zu\n", str, len);

  uint32_t rounds = argc == 3 ? atoi(argv[2]) : 10;
  uint32_t mem = argc == 4 ? atoi(argv[3]) : 10;
  uint32_t threads =
      argc == 5 ? atoi(argv[4]) : std::thread::hardware_concurrency();

  find_pubkey(str, len, rounds, mem, threads);

  return 0;
}
