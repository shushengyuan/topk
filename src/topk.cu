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
#include "topk.h"

typedef uint4 group_t;  // uint32_t
#define CHECK(res)          \
  if (res != cudaSuccess) { \
    exit(-1);               \
  }
#define GROUP_SIZE 8

dim3 numBlocks(32, 32);
dim3 threadsPerBlock(32, 32);

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const uint16_t *doc_lens,
    const size_t n_docs, const uint16_t *query, const int query_len,
    float *scores, int *d_index, size_t doc_size) {
  // each thread process one doc-query pair scoring task
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                tnum = gridDim.x * blockDim.x;

  if (tid >= n_docs) {
    return;
  }

  __shared__ uint32_t query_on_shm[MAX_QUERY_SIZE];

#pragma unroll
  for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
    query_on_shm[i] = query[i];  // 不太高效的查询加载，假设它不是热点
  }

  __syncthreads();

  for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
    register int query_idx = 0;

    register float tmp_score = 0.;

    register bool no_more_load = false;

    register size_t doc_len = n_docs >> 3;

    for (auto i = 0; i < doc_len; i++) {
      if (no_more_load) {
        break;
      }
      register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id];  // tid
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < GROUP_SIZE; j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        int left = query_idx;
        int right = query_len - 1;
        int mid;
        while (left <= right) {
          mid = (left + right) >> 1;
          if (query_on_shm[mid] < doc_segment[j]) {
            left = mid + 1;
          } else {
            right = mid - 1;
          }
        }
        query_idx = left;  // update the query index

        if (query_idx < query_len) {
          tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
        }
      }
      __syncwarp();
    }
    scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]);  // tid
    d_index[doc_id] = doc_id;
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
  register auto tidy = blockIdx.y * blockDim.y + threadIdx.y,
                tnumy = gridDim.y * blockDim.y;
#pragma unroll
  for (auto i = tidx; i < n_docs; i += tnumx) {
    // register auto layer_1_offset = i;
    register auto layer_1_total_offset = i << 3;
    register auto base_id = d_doc_sum[i];
    register auto d_lens = d_doc_lens[i];
#pragma unroll
    for (auto j = tidy; j < d_lens; j += tnumy) {
      register auto layer_0_offset = j >> 3;  // group_sz;
      register auto layer_2_offset = j & 7;   // j % group_sz;
      register auto final_offset = layer_0_offset * layer_0_stride +
                                   layer_1_total_offset + layer_2_offset;
      d_docs[final_offset] = temp_docs[base_id + j];
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
  cudaMalloc(d_sort_scores, sizeof(float) * n_docs);
  CHECK(cudaMalloc(s_indices, sizeof(int) * n_docs));
}
void d_sort_index_malloc(int **d_sort_index, float **d_scores, size_t n_docs) {
  // cudaSetDevice(0);
  CHECK(cudaMalloc(d_scores, sizeof(float) * n_docs));
  cudaMalloc(d_sort_index, sizeof(int) * n_docs);
}

void temp_docs_copy(uint16_t **temp_docs, uint16_t *h_docs,
                    uint32_t *h_docs_vec, size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(temp_docs, sizeof(uint16_t) * h_docs_vec[n_docs]);
  CHECK(cudaMemcpy(*temp_docs, h_docs, sizeof(uint16_t) * h_docs_vec[n_docs],
                   cudaMemcpyHostToDevice));
}

void d_doc_lens_malloc(uint16_t **d_doc_lens, std::vector<uint16_t> &lens,
                       size_t n_docs) {
  // cudaSetDevice(0);
  cudaMalloc(d_doc_lens, sizeof(uint16_t) * n_docs);
  CHECK(cudaMemcpy(*d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs,
                   cudaMemcpyHostToDevice));
}
void d_doc_sum_copy(uint32_t **d_doc_sum, uint32_t *h_docs_vec, size_t n_docs) {
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
  std::thread malloc_thread_5(d_doc_sum_copy, &d_doc_sum, h_docs_vec, n_docs);

  uint16_t *h_docs = new uint16_t[doc_size * n_docs];
  size_t num_threads = 10;
  std::vector<std::thread> threads(num_threads);
  std::vector<std::thread> s_threads(querys_len);

  register size_t chunk_size = n_docs / num_threads;  // 分块大小
  for (size_t i = 0; i < num_threads; i++) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? n_docs : start + chunk_size;
    threads[i] = std::thread(pre_process, std::ref(docs), h_docs, h_docs_vec,
                             start, end);
  }

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
  CHECK(cudaMallocAsync(&d_query, sizeof(uint16_t) * max_query, streams[0]));
  // std::chrono::high_resolution_clock::time_point t4 =
  //     std::chrono::high_resolution_clock::now();
  malloc_thread_2.join();
  malloc_thread_3.join();
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_scores, d_sort_scores, s_indices,
                                            d_sort_index, n_docs);
  CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, streams[0]));

  malloc_thread_1.join();
  malloc_thread_4.join();
  malloc_thread_5.join();
  copy_thread_1.join();

  // std::chrono::high_resolution_clock::time_point t5 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "first join cost "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t5 -
  //     t4).count()
  //     << " ms " << std::endl;

  // std::chrono::high_resolution_clock::time_point t5 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "before pre_process_global cost "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t5 -
  //     t1).count()
  //     << " ms " << std::endl;
  pre_process_global<<<numBlocks, threadsPerBlock>>>(
      temp_docs, d_docs, d_doc_lens, n_docs, d_doc_sum);
  // std::chrono::high_resolution_clock::time_point t6 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "init cost "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t6 -
  //     t1).count()
  //     << " ms " << std::endl;
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
        d_temp_storage, temp_storage_bytes, d_scores, d_sort_scores, s_indices,
        d_sort_index, n_docs);
    // nvtxRangePop();
    CHECK(cudaMemcpyAsync(indices_pre[i].data(), d_sort_index,
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
  CHECK(cudaFreeAsync(d_doc_lens, streams[5]));
  CHECK(cudaFreeAsync(temp_docs, streams[6]));
  free(h_docs);
}