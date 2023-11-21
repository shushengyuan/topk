#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "unistd.h"

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define CHECK(err) (HandleError(err, __FILE__, __LINE__))
