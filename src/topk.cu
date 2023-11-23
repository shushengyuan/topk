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

typedef ulong4 group_t;  // uint32_t

#define GROUP_SIZE 16  // ulong4: 16
#define SHIFT_SIZE 4

__launch_bounds__(128, 1) void __global__
    docQueryScoringCoalescedMemoryAccessSampleKernel(
        const __restrict__ uint16_t *docs,
        const __restrict__ uint16_t *doc_lens, const size_t n_docs,
        const __restrict__ uint16_t *query, const int query_len, float *scores,
        int *d_index, const size_t doc_size) {
  // each thread process one doc-query pair scoring task
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                tnum = gridDim.x * blockDim.x;
  register auto tidx = threadIdx.x;

  if (tid >= n_docs) {
    return;
  }

  __shared__ uint32_t query_on_shm[MAX_QUERY_SIZE];

  if (tidx < query_len) {
    query_on_shm[tidx] = query[tidx];  // 不太高效的查询加载，假设它不是热点
    query_on_shm[tidx + 64] = tidx + 64 < query_len ? query[tidx + 64] : 0;
  }

  register int query_idx = 0;
  register float tmp_score = 0.;
  register bool no_more_load = false;
  register int right;
  register int mid;
  register size_t doc_len =
      doc_size >>
      SHIFT_SIZE;  // MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t));

  __syncthreads();

  for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
    query_idx = 0;
    tmp_score = 0.;
    no_more_load = false;
    register group_t *docs_register = (group_t *)docs + doc_id;

    for (auto i = 0; i < doc_len && !no_more_load; i++) {
      register group_t loaded = docs_register[i * n_docs];  // tid
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < GROUP_SIZE; j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
        }
        // int left = query_idx;
        right = query_len - 1;
        while (query_idx <= right) {
          mid = (query_idx + right) >> 1;
          if (query_on_shm[mid] < doc_segment[j]) {
            query_idx = mid + 1;
          } else {
            right = mid - 1;
          }
        }
        // query_idx = left;  // update the query index

        tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
      }
      // __syncwarp();
    }
    scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]);  // tid
    d_index[doc_id] = doc_id;
  }
}

__launch_bounds__(32, 1) __global__
    void pre_process_global(const __restrict__ uint16_t *temp_docs,
                            uint16_t *d_docs,
                            const __restrict__ uint16_t *d_doc_lens,
                            const size_t n_docs,
                            const __restrict__ uint32_t *d_doc_sum) {
  // register auto group_sz = 8;  // sizeof(group_t) / sizeof(uint16_t)
  register auto layer_0_stride = n_docs << SHIFT_SIZE;  // group_sz;
  // register auto layer_1_stride = 8;           // group_sz;

  register auto tidx = blockIdx.x * blockDim.x + threadIdx.x,
                tnumx = gridDim.x * blockDim.x;
#pragma unroll
  for (auto i = tidx; i < n_docs; i += tnumx) {
    // register auto layer_1_offset = i;
    register auto layer_1_total_offset = i << SHIFT_SIZE;
    register auto base_id = d_doc_sum[i];
    register auto d_lens = d_doc_lens[i];
    register auto temp_docs_register = temp_docs + base_id;
#pragma unroll
    for (auto j = 0; j < d_lens; j++) {
      register auto layer_0_offset = j >> SHIFT_SIZE;       // group_sz;
      register auto layer_2_offset = j & (GROUP_SIZE - 1);  // j % group_sz;
      register auto final_offset = layer_0_offset * layer_0_stride +
                                   layer_1_total_offset + layer_2_offset;
      d_docs[final_offset] = temp_docs_register[j];
    }
  }
}

void pre_process(const std::vector<std::vector<uint16_t>> &docs,
                 uint16_t *h_docs, const uint32_t *h_docs_vec,
                 const size_t start_idx, const size_t lens) {
// h_docs_vec[0] = 0;
#pragma unroll
  for (size_t i = start_idx; i < lens; i++) {
    auto doc_size = docs[i].size();
    memcpy(h_docs + h_docs_vec[i], &docs[i][0], doc_size * sizeof(uint16_t));
  }
}

void prepare_1(uint32_t **h_docs_vec, const std::vector<uint16_t> &lens,
               size_t *doc_size, const size_t n_docs) {
  auto it = max_element(std::begin(lens), std::end(lens));
  *doc_size = ((*it + GROUP_SIZE - 1) >> SHIFT_SIZE) << SHIFT_SIZE;
  *h_docs_vec = new uint32_t[n_docs + 1];
  std::copy(lens.begin(), lens.end(), *h_docs_vec + 1);
  std::partial_sum(*h_docs_vec + 1, *h_docs_vec + n_docs + 1, *h_docs_vec + 1);
}
void prepare_2(const std::vector<std::vector<uint16_t>> &querys,
               uint16_t *max_query) {
  for (size_t i = 0; i < querys.size(); i++) {
    auto it = max_element(std::begin(querys[i]), std::end(querys[i]));
    *max_query = max((*it), *max_query);
  }
}

void d_docs_malloc(uint16_t **d_docs, const size_t n_docs,
                   const size_t doc_size) {
  // cudaSetDevice(0);
  cudaMalloc(d_docs, sizeof(uint16_t) * doc_size * n_docs);
}
void d_sort_scores_malloc(float **d_scores, const size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(d_scores, sizeof(float) * n_docs * 2);
  // CHECK(cudaMalloc(d_sort_scores, sizeof(float) * n_docs));
}
void d_sort_index_malloc(int **s_indices, const size_t n_docs) {
  // cudaSetDevice(0);
  CHECK(cudaMalloc(s_indices, sizeof(int) * n_docs * 2));
  // cudaMalloc(d_sort_index, sizeof(int) * n_docs);
}

void temp_docs_copy(uint16_t **temp_docs, const uint16_t *h_docs,
                    const uint32_t *h_docs_vec, const size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(temp_docs, sizeof(uint16_t) * h_docs_vec[n_docs]);
  CHECK(cudaMemcpy(*temp_docs, h_docs, sizeof(uint16_t) * h_docs_vec[n_docs],
                   cudaMemcpyHostToDevice));
  // free(h_docs);
}

void d_doc_lens_malloc(uint16_t **d_doc_lens, const std::vector<uint16_t> &lens,
                       const size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(d_doc_lens, sizeof(uint16_t) * n_docs);
  CHECK(cudaMemcpy(*d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs,
                   cudaMemcpyHostToDevice));
}
void d_doc_sum_copy(uint32_t **d_doc_sum, const uint32_t *h_docs_vec,
                    const size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(d_doc_sum, sizeof(uint32_t) * (n_docs + 1));

  CHECK(cudaMemcpy(*d_doc_sum, h_docs_vec, sizeof(uint32_t) * (n_docs + 1),
                   cudaMemcpyHostToDevice));
}

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices  // shape [querys.size(), TOPK]
) {
  // std::chrono::high_resolution_clock::time_point t1 =
  //     std::chrono::high_resolution_clock::now();
  constexpr int block = N_THREADS_IN_ONE_BLOCK;
  constexpr int block_pre = 16;
  register size_t n_docs = docs.size();
  int grid = (n_docs + block - 1) / block;
  int grid_pre = (n_docs + block_pre - 1) / block_pre;
  uint16_t querys_len = querys.size();

  // int *d_sort_index = nullptr;
  // float *d_sort_scores = nullptr;
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
  // printf("(sizeof(group_t) / sizeof(uint16_t) %ld \n",
  //        (sizeof(ulong4) / sizeof(uint16_t)));

  std::thread prepare_thread_1(prepare_1, &h_docs_vec, std::ref(lens),
                               &doc_size, n_docs);
  std::thread prepare_thread_2(prepare_2, std::ref(querys), &max_query);

  std::thread malloc_thread_2(d_sort_scores_malloc, &d_scores, n_docs);
  std::thread malloc_thread_3(d_sort_index_malloc, &s_indices, n_docs);
  std::thread malloc_thread_4(d_doc_lens_malloc, &d_doc_lens, std::ref(lens),
                              n_docs);
  prepare_thread_1.join();
  // printf("%ld \n",  doc_size / sizeof(group_t) / sizeof(uint16_t));

  std::thread malloc_thread_1(d_docs_malloc, &d_docs, n_docs, doc_size);
  std::thread malloc_thread_5(d_doc_sum_copy, &d_doc_sum, h_docs_vec, n_docs);

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
  streams = (cudaStream_t *)malloc(querys_len * sizeof(cudaStream_t));
  for (int i = 0; i < querys_len; ++i) {
    CHECK(cudaStreamCreate(&streams[i]));
  }
  for (std::thread &t : threads) {
    t.join();  // 等待所有线程完成
  }

  std::thread copy_thread_1(temp_docs_copy, &temp_docs, h_docs, h_docs_vec,
                            n_docs);

  prepare_thread_2.join();
  CHECK(cudaMallocAsync(&d_query, sizeof(uint16_t) * max_query, streams[0]));

  malloc_thread_2.join();
  malloc_thread_3.join();
  cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_scores, d_scores + n_docs,
      s_indices, s_indices + n_docs, n_docs, 0, sizeof(float) * 8, streams[1]);
  CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, streams[1]));

  malloc_thread_1.join();
  malloc_thread_4.join();
  malloc_thread_5.join();
  copy_thread_1.join();

  pre_process_global<<<grid_pre, block_pre, 0, streams[0]>>>(
      temp_docs, d_docs, d_doc_lens, n_docs, d_doc_sum);
  CHECK(cudaFreeAsync(temp_docs, streams[querys_len - 1]));
  CHECK(cudaFreeAsync(d_doc_sum, streams[querys_len - 2]));
  CHECK(cudaFreeAsync(d_doc_lens, streams[querys_len - 3]));

  // std::chrono::high_resolution_clock::time_point t6 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "init cost "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t6 -
  //     t1).count()
  //     << " ms " << std::endl;
  cudaStreamSynchronize(streams[1]);
  for (int i = 0; i < querys_len; ++i) {
    auto &query = querys[i];
    const size_t query_len = query.size();
    // nvtxRangePushA("cuda malloc");
    CHECK(cudaMemcpyAsync(d_query, query.data(), sizeof(uint16_t) * query_len,
                          cudaMemcpyHostToDevice, streams[i]));

    docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0,
                                                       streams[i]>>>(
        d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores, s_indices,
        doc_size);

    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_scores, d_scores + n_docs,
        s_indices, s_indices + n_docs, n_docs, 0, sizeof(float) * 8,
        streams[i]);
    // nvtxRangePop();
    CHECK(cudaMemcpyAsync(indices_pre[i].data(), s_indices + n_docs,
                          sizeof(int) * TOPK, cudaMemcpyDeviceToHost,
                          streams[i]));
  }
  indices = indices_pre;
  // deallocation
  CHECK(cudaFreeAsync(d_scores, streams[0]));
  CHECK(cudaFreeAsync(s_indices, streams[1]));
  CHECK(cudaFreeAsync(d_query, streams[2]));
  CHECK(cudaFreeAsync(d_temp_storage, streams[3]));
  CHECK(cudaFreeAsync(d_docs, streams[4]));

  // CHECK(cudaFreeAsync(d_sort_index, streams[6]));

  // free(h_docs);
}