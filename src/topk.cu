#include <nvtx3/nvToolsExt.h>
// #include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

// #include <numeric>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <numeric>
#include <thread>

#include "assert.h"
#include "common.h"
#include "topk.h"

typedef uint4 group_t;  // uint32_t

#define GROUP_SIZE 8

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const uint16_t *doc_lens,
    const size_t n_docs, const uint16_t *query, const int query_len,
    const int query_len_split, float *scores, int *d_index, size_t doc_size,
    size_t max_query) {
  // each thread process one doc-query pair scoring task
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                tnum = gridDim.x * blockDim.x;

  if (tid >= n_docs) {
    return;
  }

  __shared__ uint32_t query_on_shm_1[MAX_QUERY_SIZE];
  __shared__ uint32_t query_on_shm_2[MAX_QUERY_SIZE];

#pragma unroll
  for (auto i = threadIdx.x; i < query_len_split; i += blockDim.x) {
    query_on_shm_1[i] = query[i];  // 不太高效的查询加载，假设它不是热点
  }
#pragma unroll
  for (auto i = threadIdx.x; i < query_len - query_len_split; i += blockDim.x) {
    query_on_shm_2[i] =
        query[i + max_query];  // 不太高效的查询加载，假设它不是热点
  }
  __syncthreads();
  for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
    register int query_idx_1 = 0;
    register int query_idx_2 = query_len_split;

    register float tmp_score_1 = 0.;
    register float tmp_score_2 = 0.;

    register bool no_more_load = false;
    register size_t doc_len = n_docs >> 3;
    // register group_t *docs_register = (group_t *)docs + doc_id;

    __syncthreads();

    for (auto i = 0; i < doc_len; i++) {
      if (no_more_load) {
        break;
      }
      register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id];  // tidx
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < GROUP_SIZE; j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        // int left = query_idx_1;
        int right_1 = query_len_split - 1;
        int mid_1;
        while (query_idx_1 <= right_1) {
          mid_1 = (query_idx_1 + right_1) >> 1;
          if (query_on_shm_1[mid_1] < doc_segment[j]) {
            query_idx_1 = mid_1 + 1;
          } else {
            right_1 = mid_1 - 1;
          }
        }
        int right_2 = query_len - 1;
        int mid_2;
        while (query_idx_2 <= right_2) {
          mid_2 = (query_idx_2 + right_2) >> 1;
          if (query_on_shm_2[mid_2 - query_len_split] < doc_segment[j]) {
            query_idx_2 = mid_2 + 1;
          } else {
            right_2 = mid_2 - 1;
          }
        }
        tmp_score_1 += (query_on_shm_1[query_idx_1] == doc_segment[j]);
        tmp_score_2 +=
            (query_on_shm_2[query_idx_2 - query_len_split] == doc_segment[j]);
      }
      __syncwarp();
    }

    scores[doc_id] =
        tmp_score_1 / max(query_len_split, doc_lens[doc_id]);  // tidx
    scores[n_docs + doc_id] = tmp_score_2 / max(query_len - query_len_split,
                                                doc_lens[doc_id]);  // tidx
    d_index[doc_id] = doc_id;
    d_index[n_docs + doc_id] = doc_id;
  }
}
__global__ void pre_process_global(const uint16_t *temp_docs, uint16_t *d_docs,
                                   const uint16_t *d_doc_lens,
                                   const size_t n_docs,
                                   const uint32_t *d_doc_sum) {
  // register auto group_sz = 8;  // sizeof(group_t) / sizeof(uint16_t)
  register auto layer_0_stride = n_docs * 8;  // group_sz;
  // register auto layer_1_stride = 8;           // group_sz;

  register auto tidx = blockIdx.x * blockDim.x + threadIdx.x,
                tnumx = gridDim.x * blockDim.x;
#pragma unroll
  for (auto i = tidx; i < n_docs; i += tnumx) {
    // register auto layer_1_offset = i;
    register auto layer_1_total_offset = i << 3;
    register auto base_id = d_doc_sum[i];
    register auto d_lens = d_doc_lens[i];
    register auto temp_docs_register = temp_docs + base_id;
#pragma unroll
    for (auto j = 0; j < d_lens; j++) {
      register auto layer_0_offset = j >> 3;  // group_sz;
      register auto layer_2_offset = j & 7;   // j % group_sz;
      register auto final_offset = layer_0_offset * layer_0_stride +
                                   layer_1_total_offset + layer_2_offset;
      d_docs[final_offset] = temp_docs_register[j];
    }
  }
}

void pre_process(std::vector<std::vector<uint16_t>> &docs, uint16_t *h_docs,
                 uint32_t *h_docs_vec, size_t start_idx, size_t lens) {
// h_docs_vec[0] = 0;
#pragma unroll
  for (size_t i = start_idx; i < lens; i++) {
    auto doc_size = docs[i].size();
    memcpy(h_docs + h_docs_vec[i], &docs[i][0], doc_size * sizeof(uint16_t));
  }
}

void prepare_1(uint32_t **h_docs_vec, std::vector<uint16_t> &lens,
               size_t *doc_size, size_t n_docs) {
  auto it = max_element(std::begin(lens), std::end(lens));
  *doc_size = (*it + 8 >> 3) << 3;
  *h_docs_vec = new uint32_t[n_docs + 1];
  std::copy(lens.begin(), lens.end(), *h_docs_vec + 1);
  std::partial_sum(*h_docs_vec + 1, *h_docs_vec + n_docs + 1, *h_docs_vec + 1);
}
void prepare_2(std::vector<std::vector<uint16_t>> &querys,
               uint16_t *max_query) {
  for (size_t i = 0; i < querys.size(); i++) {
    auto it = max_element(std::begin(querys[i]), std::end(querys[i]));
    *max_query = max((*it), *max_query);
  }
}

void d_docs_malloc(uint16_t **d_docs, size_t n_docs, size_t doc_size) {
  // cudaSetDevice(0);
  cudaMalloc(d_docs, sizeof(uint16_t) * doc_size * n_docs);
}
void d_sort_scores_malloc(float **d_sort_scores, int **s_indices,
                          size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(d_sort_scores, sizeof(float) * n_docs * 2);
  CHECK(cudaMalloc(s_indices, sizeof(int) * n_docs * 2));
}
void d_sort_index_malloc(int **d_sort_index, float **d_scores, size_t n_docs) {
  // cudaSetDevice(0);
  CHECK(cudaMalloc(d_scores, sizeof(float) * n_docs * 2));
  cudaMalloc(d_sort_index, sizeof(int) * n_docs * 2);
}

void temp_docs_copy(uint16_t **temp_docs, uint16_t *h_docs,
                    uint32_t *h_docs_vec, size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(temp_docs, sizeof(uint16_t) * h_docs_vec[n_docs]);
  CHECK(cudaMemcpy(*temp_docs, h_docs, sizeof(uint16_t) * h_docs_vec[n_docs],
                   cudaMemcpyHostToDevice));
  // free(h_docs);
}

void d_doc_lens_malloc(uint16_t **d_doc_lens, std::vector<uint16_t> &lens,
                       size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(d_doc_lens, sizeof(uint16_t) * n_docs);
  CHECK(cudaMemcpy(*d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs,
                   cudaMemcpyHostToDevice));
}
void d_doc_sum_copy(uint32_t **d_doc_sum, uint16_t **temp_docs,
                    uint32_t *h_docs_vec, std::vector<uint16_t> &lens,
                    size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(d_doc_sum, sizeof(uint32_t) * (n_docs + 1));

  CHECK(cudaMemcpy(*d_doc_sum, h_docs_vec, sizeof(uint32_t) * (n_docs + 1),
                   cudaMemcpyHostToDevice));
}

int block = N_THREADS_IN_ONE_BLOCK;

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices  // shape [querys.size(), TOPK]
) {
  // std::chrono::high_resolution_clock::time_point t1 =
  //     std::chrono::high_resolution_clock::now();

  register size_t n_docs = docs.size();
  int grid = (n_docs + block - 1) / block;
  int querys_len = querys.size();

  int *d_sort_index = nullptr;
  float *d_sort_scores = nullptr;
  void *d_temp_storage = nullptr;
  uint32_t *d_doc_sum = nullptr;
  uint16_t *temp_docs = nullptr;
  size_t temp_storage_bytes = 0;

  uint16_t *d_docs = nullptr;
  uint16_t *d_doc_lens = nullptr;

  uint32_t *h_docs_vec = nullptr;
  size_t doc_size = 0;
  uint16_t max_query = 0;

  float *d_scores = nullptr;
  int *s_indices = nullptr;
  uint16_t *d_query = nullptr;
  cudaStream_t *streams;

  std::thread prepare_thread_1(prepare_1, &h_docs_vec, std::ref(lens),
                               &doc_size, n_docs);
  std::thread prepare_thread_2(prepare_2, std::ref(querys), &max_query);

  std::thread malloc_thread_2(d_sort_scores_malloc, &d_sort_scores, &s_indices,
                              n_docs);
  std::thread malloc_thread_3(d_sort_index_malloc, &d_sort_index, &d_scores,
                              n_docs);
  std::thread malloc_thread_4(d_doc_lens_malloc, &d_doc_lens, std::ref(lens),
                              n_docs);
  prepare_thread_1.join();
  std::thread malloc_thread_1(d_docs_malloc, &d_docs, n_docs, doc_size);
  std::thread malloc_thread_5(d_doc_sum_copy, &d_doc_sum, &temp_docs,
                              h_docs_vec, std::ref(lens), n_docs);

  uint16_t *h_docs = new uint16_t[doc_size * n_docs];
  size_t num_threads = 10;
  std::vector<std::thread> threads(num_threads);
  std::vector<std::thread> s_threads(querys_len);

  register size_t chunk_size = n_docs / num_threads;  // 分块大小
  for (size_t i = 0; i < num_threads - 1; i++) {
    size_t start = i * chunk_size;
    size_t end = start + chunk_size;
    threads[i] = std::thread(pre_process, std::ref(docs), h_docs, h_docs_vec,
                             start, end);
  }
  threads[num_threads - 1] =
      std::thread(pre_process, std::ref(docs), h_docs, h_docs_vec,
                  (num_threads - 1) * chunk_size, n_docs);

  std::vector<std::vector<int>> indices_pre(querys_len, std::vector<int>(TOPK));

  for (std::thread &t : threads) {
    t.join();  // 等待所有线程完成
  }

  std::thread copy_thread_1(temp_docs_copy, &temp_docs, h_docs, h_docs_vec,
                            n_docs);

  streams = (cudaStream_t *)malloc(querys_len * sizeof(cudaStream_t));
  for (int i = 0; i < querys_len; ++i) {
    CHECK(cudaStreamCreate(&streams[i]));
  }

  prepare_thread_2.join();
  CHECK(
      cudaMallocAsync(&d_query, sizeof(uint16_t) * max_query * 2, streams[0]));

  malloc_thread_2.join();
  malloc_thread_3.join();
  cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_scores, d_sort_scores, s_indices,
      d_sort_index, n_docs, 0, sizeof(float) * 8, streams[0]);
  CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, streams[0]));

  malloc_thread_1.join();
  malloc_thread_4.join();
  malloc_thread_5.join();
  copy_thread_1.join();

  pre_process_global<<<grid, block>>>(temp_docs, d_docs, d_doc_lens, n_docs,
                                      d_doc_sum);
  CHECK(cudaFreeAsync(temp_docs, streams[6]));

  // std::chrono::high_resolution_clock::time_point t6 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "init cost "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t6 -
  //     t1).count()
  //     << " ms " << std::endl;

  for (int i = 0; i < querys_len / 2; ++i) {
    auto &query_1 = querys[i * 2];
    auto &query_2 = querys[i * 2 + 1];
    const size_t query_len_1 = query_1.size();
    const size_t query_len_2 = query_2.size();
    const size_t q_sum = query_len_1 + query_len_2;
    // nvtxRangePushA("cuda malloc");
    CHECK(cudaMemcpyAsync(d_query, query_1.data(),
                          sizeof(uint16_t) * (query_len_1),
                          cudaMemcpyHostToDevice, streams[i]));
    CHECK(cudaMemcpyAsync(d_query + max_query, query_2.data(),
                          sizeof(uint16_t) * (query_len_2),
                          cudaMemcpyHostToDevice, streams[i]));

    docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                       streams[i]>>>(
        d_docs, d_doc_lens, n_docs, d_query, q_sum, query_len_1, d_scores,
        s_indices, doc_size, max_query);

    CHECK(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores, d_sort_scores, s_indices,
        d_sort_index, n_docs, 0, sizeof(float) * 8, streams[i]));
    CHECK(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores + n_docs,
        d_sort_scores + n_docs, s_indices + n_docs, d_sort_index + n_docs,
        n_docs, 0, sizeof(float) * 8, streams[i]));
    // nvtxRangePop();
    // printf("12 \n");
    CHECK(cudaMemcpyAsync(indices_pre[i * 2].data(), d_sort_index,
                          sizeof(int) * TOPK, cudaMemcpyDeviceToHost,
                          streams[i]));
    CHECK(cudaMemcpyAsync(indices_pre[i * 2 + 1].data(), d_sort_index + n_docs,
                          sizeof(int) * TOPK, cudaMemcpyDeviceToHost,
                          streams[i]));
  }
  if (querys_len / 2 * 2 != querys_len) {
    auto &query_1 = querys[querys_len - 1];
    const size_t query_len_1 = query_1.size();
    // nvtxRangePushA("cuda malloc");
    CHECK(cudaMemcpyAsync(d_query, query_1.data(),
                          sizeof(uint16_t) * (query_len_1),
                          cudaMemcpyHostToDevice, streams[querys_len - 1]));

    docQueryScoringCoalescedMemoryAccessSampleKernel<<<
        grid, block, 0, streams[querys_len - 1]>>>(
        d_docs, d_doc_lens, n_docs, d_query, query_len_1, query_len_1, d_scores,
        s_indices, doc_size, max_query);

    CHECK(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores, d_sort_scores, s_indices,
        d_sort_index, n_docs, 0, sizeof(float) * 8, streams[querys_len - 1]));
    CHECK(cudaMemcpyAsync(indices_pre[querys_len - 1].data(), d_sort_index,
                          sizeof(int) * TOPK, cudaMemcpyDeviceToHost,
                          streams[querys_len - 1]));
  }
  indices = indices_pre;
  // deallocation
  CHECK(cudaFreeAsync(d_scores, streams[0]));
  CHECK(cudaFreeAsync(s_indices, streams[1]));
  CHECK(cudaFreeAsync(d_query, streams[2]));
  CHECK(cudaFreeAsync(d_temp_storage, streams[3]));
  CHECK(cudaFreeAsync(d_docs, streams[4]));
  CHECK(cudaFreeAsync(d_doc_lens, streams[5]));
  CHECK(cudaFreeAsync(d_sort_index, streams[6]));

  // free(h_docs);
}