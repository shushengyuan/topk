#pragma once
#include <cuda.h>

#include <algorithm>
#include <iostream>
#include <vector>

#define MAX_DOC_SIZE 128

void pre_process(std::vector<std::vector<uint16_t>> &docs, uint16_t *h_docs,
                 std::vector<int> &h_doc_lens_vec);