#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "unistd.h"

#define MAX_DOC_SIZE 128
#define MAX_QUERY_SIZE 128
#define N_THREADS_IN_ONE_BLOCK 64
#define TOPK 100
#define GROUP_SIZE 8

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &query,
                                    std::vector<std::vector<uint16_t>> &docs,
                                    std::vector<uint16_t> &lens,
                                    std::vector<std::vector<int>> &indices);
