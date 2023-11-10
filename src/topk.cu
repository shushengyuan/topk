#include <nvtx3/nvToolsExt.h>
// #include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <thread>

#include "assert.h"
#include "topk.h"

typedef uint4 group_t;  // uint32_t
#define CHECK(res)          \
  if (res != cudaSuccess) { \
    exit(-1);               \
  }

// device A100
//  cpu sort :
//  yuan trust sort L: 3002 ms
//  yuan trust sort L: 2750 ms

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const uint16_t *doc_lens,
    const size_t n_docs, uint16_t *query, const int total_querys_len,
    const uint16_t *query_lens_d, float *scores, int *s_indices) {
  // each thread process one doc-query pair scoring task
  register auto tidx = blockIdx.x * blockDim.x + threadIdx.x,
                tnumx = gridDim.x * blockDim.x;

  if (tidx >= n_docs * total_querys_len) {
    return;
  }
  __shared__ uint32_t query_on_shm[MAX_QUERY_SIZE];
  // __shared__ uint32_t doc_lens_on_shm[n_docs];

  for (auto doc_id = tidx; doc_id < n_docs * total_querys_len;
       doc_id += tnumx) {
    register int query_idx = 0;

    register float tmp_score = 0.;

    register bool no_more_load = false;
    register auto query_len = query_lens_d[doc_id / n_docs];
    register auto start_index = (doc_id / n_docs) * MAX_QUERY_SIZE;
    register auto doc_index = doc_id % n_docs;

#pragma unroll
    for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
      query_on_shm[i] =
          query[start_index + i];  // not very efficient query loading
                                   // temporally, as assuming its not hotspot
    }
    __syncthreads();

    for (auto i = 0; i < MAX_DOC_SIZE / GROUP_SIZE; i++) {
      if (no_more_load) {
        break;
      }
      register group_t loaded =
          ((group_t *)docs)[i * n_docs + doc_index];  // tidx
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < GROUP_SIZE; j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        while (query_idx < query_len &&
               query_on_shm[query_idx] < doc_segment[j]) {
          ++query_idx;
        }

        if (query_idx < query_len) {
          tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
        }
      }
      __syncwarp();
    }
    scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_index]);  // tidx
    s_indices[doc_id] = doc_index;
  }
}
__global__ void pre_process_global(const uint16_t *temp_docs, uint16_t *d_docs,
                                   const uint16_t *d_doc_lens,
                                   const size_t n_docs) {
  register auto group_sz = sizeof(group_t) / sizeof(uint16_t);
  register auto layer_0_stride = n_docs * group_sz;
  register auto layer_1_stride = group_sz;

  register auto tidx = blockIdx.x * blockDim.x + threadIdx.x,
                tnumx = gridDim.x * blockDim.x;
  register auto tidy = blockIdx.y * blockDim.y + threadIdx.y,
                tnumy = gridDim.y * blockDim.y;

  for (auto i = tidx; i < n_docs; i += tnumx) {
    register auto layer_1_offset = i;
    register auto layer_1_total_offset = layer_1_offset * layer_1_stride;
    for (auto j = tidy; j < d_doc_lens[i]; j += tnumy) {
      register auto layer_0_offset = j / group_sz;
      register auto layer_2_offset = j % group_sz;
      register auto final_offset = layer_0_offset * layer_0_stride +
                                   layer_1_total_offset + layer_2_offset;
      d_docs[final_offset] = temp_docs[i * MAX_DOC_SIZE + j];
    }
  }
}

void pre_process(std::vector<std::vector<uint16_t>> &docs, uint16_t *h_docs) {
  // h_docs_sum[0] = 0 ;
#pragma unroll
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc_size = docs[i].size();
    // if (i != 0) {
    //   h_docs_sum[i] = h_docs_sum[i - 1] + doc_size;
    // }
    for (size_t j = 0; j < doc_size; j++) {
      h_docs[i * MAX_DOC_SIZE + j] = docs[i][j];
    }
  }
}
void query_thread(std::vector<std::vector<uint16_t>> &querys, uint16_t *h_query,
                  std::vector<uint16_t> &query_lens_vec) {
#pragma unroll
  for (size_t i = 0; i < querys.size(); i++) {
    auto querys_size = querys[i].size();
    query_lens_vec[i] = querys_size;
    for (size_t j = 0; j < querys_size; j++) {
      h_query[i * MAX_QUERY_SIZE + j] = querys[i][j];
    }
  }
}

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices  // shape [querys.size(), TOPK]
) {
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  size_t n_docs = docs.size();
  int total_querys_len = querys.size();

  int block = N_THREADS_IN_ONE_BLOCK;
  int grid = ((total_querys_len * n_docs) + block - 1) / block;

  uint16_t *d_docs = nullptr;
  uint16_t *d_doc_lens = nullptr;
  uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
  uint16_t *h_query = new uint16_t[MAX_QUERY_SIZE * total_querys_len];
  uint16_t *temp_docs = nullptr;
  std::vector<uint16_t> query_lens_vec(total_querys_len);
  std::vector<std::vector<int>> indices_pre(querys.size(),
                                            std::vector<int>(TOPK));
  std::thread t_pre(pre_process, std::ref(docs), h_docs);
  std::thread t_query(query_thread, std::ref(querys), h_query,
                      std::ref(query_lens_vec));

  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);
  cudaSetDevice(0);

  dim3 numBlocks(32, 32);
  dim3 threadsPerBlock(32, 32);

  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(total_querys_len * sizeof(cudaStream_t));
  for (int i = 0; i < total_querys_len; i++) {
    cudaStreamCreate(&streams[i]);
  }

  cudaMallocAsync(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
                  streams[0]);
  cudaMallocAsync(&temp_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
                  streams[2]);
  cudaMallocAsync(&d_doc_lens, sizeof(uint16_t) * n_docs, streams[1]);
  cudaMemcpyAsync(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs,
                  cudaMemcpyHostToDevice, streams[1]);
  cudaStreamSynchronize(streams[1]);
  t_pre.join();
  cudaMemcpyAsync(temp_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
                  cudaMemcpyHostToDevice, streams[2]);

  cudaStreamSynchronize(streams[2]);
  nvtxRangePushA("pre_process_global start");
  pre_process_global<<<numBlocks, threadsPerBlock, 0, streams[0]>>>(
      temp_docs, d_docs, d_doc_lens, n_docs);
  nvtxRangePop();
  cudaStreamSynchronize(streams[0]);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();

  std::cout
      << "init cost "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms " << std::endl;
  // init indices
  nvtxRangePushA("query start");
  uint16_t *d_query = nullptr;
  float *d_scores = nullptr;
  int *s_indices = nullptr;
  uint16_t *query_lens_d = nullptr;

  cudaMallocAsync(&d_query,
                  sizeof(uint16_t) * MAX_QUERY_SIZE * total_querys_len,
                  streams[0]);
  cudaMallocAsync(&query_lens_d, sizeof(uint16_t) * total_querys_len,
                  streams[1]);
  t_query.join();
  cudaMemcpyAsync(d_query, h_query,
                  sizeof(uint16_t) * MAX_QUERY_SIZE * total_querys_len,
                  cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(query_lens_d, query_lens_vec.data(),
                  sizeof(uint16_t) * total_querys_len, cudaMemcpyHostToDevice,
                  streams[1]);

  cudaMallocAsync(&d_scores, sizeof(float) * n_docs * total_querys_len,
                  streams[2]);
  cudaMallocAsync(&s_indices, sizeof(int) * n_docs * total_querys_len,
                  streams[3]);
  cudaStreamSynchronize(streams[1]);
  cudaStreamSynchronize(streams[2]);
  cudaStreamSynchronize(streams[3]);
  docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                     streams[0]>>>(
      d_docs, d_doc_lens, n_docs, d_query, total_querys_len, query_lens_d,
      d_scores, s_indices);

  nvtxRangePop();
#pragma unroll
  for (int i = 0; i < total_querys_len; ++i) {
    nvtxRangePushA("sort_by_key");
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int *d_sort_index = nullptr;
    float *d_sort_scores = nullptr;

    cudaMallocAsync(&d_sort_index, sizeof(int) * n_docs, streams[i]);
    cudaMallocAsync(&d_sort_scores, sizeof(float) * n_docs, streams[i]);

    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores + i * n_docs,
        d_sort_scores, s_indices + i * n_docs, d_sort_index, n_docs);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // cudaDeviceSynchronize();

    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores + i * n_docs,
        d_sort_scores, s_indices + i * n_docs, d_sort_index, n_docs);

    cudaMemcpyAsync(indices_pre[i].data(), d_sort_index, sizeof(int) * TOPK,
                    cudaMemcpyDeviceToHost, streams[i]);
    nvtxRangePop();
  }

  indices = indices_pre;
  // cudaFreeAsync(s_indices, streams[0]);
  // cudaFreeAsync(d_scores, streams[0]);
  // cudaFreeAsync(d_query, streams[0]);
  // // deallocation
  // cudaFree(d_docs);
  // cudaFree(d_scores);
  // cudaFree(d_doc_lens);
  // free(h_docs);
}
