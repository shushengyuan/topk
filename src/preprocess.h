#pragma once
#include <cuda.h>

#include <algorithm>
#include <iostream>
#include <vector>

#define MAX_DOC_SIZE 128

void pre_process(std::vector<std::vector<uint16_t>> &docs, float *d_scores,
                 uint16_t *d_docs, int *d_doc_lens);