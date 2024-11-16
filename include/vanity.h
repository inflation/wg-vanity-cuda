//
// Created by inflation on 11/12/2024.
//

#pragma once

void find_pubkey(char *str, int len, int rounds, int mem,
                 int threads = std::thread::hardware_concurrency());
